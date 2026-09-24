"""
运行分类后的问答流程：Jev 判断问题类别，通用问题直接请求大模型，专业问题选择检索策略、改写
查询、调用混合检索和重排，再通过提示模板组织证据并生成回答。
该类负责协调组件和传递过滤条件、历史及流式选项；
它不负责 FAQ 表维护、文档上传或用户鉴权。
"""

import time
from typing import Iterator, List

from base.config import config
from base.logger import logger
from rag_qa.generation.llm import as_llm, build_text_chain
from rag_qa.generation.prompts import RAGPrompts
from rag_qa.query.rewriter import QueryRewriter
from rag_qa.router.classifier import QueryClassifier
from rag_qa.router.strategy import STRATEGY_TO_REWRITE_MODE, StrategySelector
from rag_qa.schema.chunk import Chunk, to_chunk
from rag_qa.schema.citation import build_citation_context, validate_answer_citations

MAX_HISTORY_TURNS = 5

class RAGSystem:
    """RAG 编排层: 分类 -> 选策略 -> 改写 -> 检索 -> 拼上下文 -> 生成."""

    def __init__(
        self,
        vector_store,
        llm,
        classifier=None,
        strategy_selector=None,
        rewriter=None,
        retrieval_k: int = None,
        candidate_m: int = None,
        filters: dict = None,
    ):
        """
        函数功能: 注入检索、模型、分类与改写组件, 并装配同步和流式 LCEL 生成链.
        :param vector_store: 检索器对象(推荐用 HybridRetriever; 也兼容老的 VectorStore)
        :param llm: 大模型调用对象(LLMClient 或 callable)
        :param classifier: 查询分类器, 为空时自动创建
        :param strategy_selector: 策略选择器, 为空时自动创建
        :param rewriter: 查询改写器, 为空时自动创建
        :param retrieval_k: 检索TopK, 默认取配置
        :param candidate_m: 最终送入上下文的文档数, 默认取配置
        :param filters: 额外的检索过滤条件(租户, 文档类型, 时间范围等), 会下推到检索引擎
        :return: None
        """
        
        self.vector_store = vector_store
        self.llm = as_llm(llm)
        self.retrieval_k = retrieval_k or config.RETRIEVAL_K
        self.candidate_m = candidate_m or config.CANDIDATE_M
        self.filters = filters

        
        self.classifier = classifier if classifier is not None else QueryClassifier()
        self.strategy_selector = strategy_selector if strategy_selector is not None else StrategySelector(self.llm)
        self.rewriter = rewriter if rewriter is not None else QueryRewriter(self.llm)

        
        self.rag_chain = build_text_chain(RAGPrompts.rag_prompt(), self.llm)
        self.rag_stream_chain = build_text_chain(RAGPrompts.rag_prompt(), self.llm, streaming=True)
        self.general_chain = build_text_chain(RAGPrompts.general_prompt(), self.llm)
        self.general_stream_chain = build_text_chain(
            RAGPrompts.general_prompt(), self.llm, streaming=True
        )

    
    def retrieve(self, query: str, source_filter: str = None, strategy: str = None) -> List[Chunk]:
        """
        函数功能: 执行检索, 返回统一结构的 Chunk 列表.
        :param query: 原始查询
        :param source_filter: 来源过滤条件(旧语义是学科), None 表示不过滤
        :param strategy: 检索策略, 为空时自动选择
        :return: Chunk 列表(已去重, 已按 candidate_m 截取)
        """
        
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        
        mode = STRATEGY_TO_REWRITE_MODE.get(strategy, "direct")
        queries = self.rewriter.rewrite(query, mode=mode)
        logger.info(f"策略 {strategy} 改写后待检索的查询: {queries}")

        
        hit_chunks = []
        for q in queries:
            docs = self.vector_store.hybrid_search_with_rerank(
                q, k=self.retrieval_k, source_filter=source_filter
            ) or []
            
            hit_chunks.extend(Chunk.from_document(doc) for doc in docs)
            logger.info(f"查询 '{q}' 检索到 {len(docs)} 个文档块")

        
        unique_chunks = self._dedup(hit_chunks)
        logger.info(f"共检索到 {len(hit_chunks)} 个块, 去重后剩 {len(unique_chunks)} 个")

        
        final_chunks = unique_chunks[: self.candidate_m]
        logger.info(f"最终选取 {len(final_chunks)} 个文档块作为上下文")
        return final_chunks

    
    def _search_once(self, query: str, source_filter: str = None):
        """
        函数功能: 调一次检索, 屏蔽新旧两套检索器接口的差异.
        :param query: 单条查询
        :param source_filter: 来源过滤(旧语义学科, 新语义文档来源)
        :return: 文档块列表(Chunk 或 Document, 由上层统一转换)
        """
        
        filters = dict(self.filters or {})
        if source_filter:
            filters.setdefault("source", source_filter)

        
        search = getattr(self.vector_store, "search", None)
        if callable(search):
            return search(query, k=self.retrieval_k, filters=filters or None)

        
        return self.vector_store.hybrid_search_with_rerank(
            query, k=self.retrieval_k, source_filter=source_filter
        )

    
    @staticmethod
    def _dedup(chunks: List[Chunk]) -> List[Chunk]:
        """
        函数功能: 按 Chunk.dedup_key 保留第一次出现的文本块并维持原排序.
        :param chunks: 多条改写查询合并后的 Chunk 列表
        :return: 去重后的 Chunk 列表
        """
        seen, result = set(), []
        for chunk in chunks:
            key = chunk.dedup_key
            if key in seen:
                continue
            seen.add(key)
            result.append(chunk)
        return result

    
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        """
        函数功能: 兼容旧接口, 执行统一检索后把 Chunk 转回 LangChain Document.
        :param query: 用户原始问题
        :param source_filter: 可选知识来源过滤条件
        :param strategy: 可选的指定检索策略
        :return: LangChain Document 列表
        """
        chunks = self.retrieve(query, source_filter=source_filter, strategy=strategy)
        return [chunk.to_document() for chunk in chunks]

    
    @staticmethod
    def _normalize_history(history):
        """
        函数功能: 校验历史格式, 非法则丢弃, 合法则只保留最近 MAX_HISTORY_TURNS 轮.
        :param history: 历史列表, 元素形如 {'question': ..., 'answer': ...}
        :return: 规范化后的历史列表
        """
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}, 忽略历史")
            return []
        if history:
            history = history[-MAX_HISTORY_TURNS:]
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}, 忽略历史")
                    return []
        return history or []

    
    def _build_rag_inputs(self, query: str, context: str, history) -> dict:
        """
        函数功能: 把问题、证据、历史和客服电话整理成 RAG LCEL 链的输入字典.
        :param query: 用户原始问题
        :param context: 带引用编号的检索证据文本
        :param history: 已规范化的最近对话列表
        :return: 与 RAGPrompts.rag_prompt 输入变量对应的字典
        """
        history_context = ""
        if history:
            history_context = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in history)
            logger.info(f"使用对话历史: {history_context[:100]}...")
        return {
            "context": context,
            "history": history_context,
            "question": query,
            "phone": config.CUSTOMER_SERVICE_PHONE,
        }

    
    def _prepare(self, query: str, source_filter: str = None, history=None):
        """
        函数功能: 分类并准备提示词, 检索阶段的逻辑集中在这里, 生成阶段(流式/非流式)各自复用.
        :param query: 用户原始问题
        :param source_filter: 可选知识来源过滤条件
        :param history: 可选会话历史列表
        :return: (chain_inputs, category, chunks) -> LCEL 链输入, 查询类别, 命中的文档块
        """
        history = self._normalize_history(history)

        
        category = self.classifier.predict_category(query)
        logger.info(f"查询分类结果: {category} (查询: '{query}')")

        if category == "通用知识":
            logger.info("查询为通用知识, 直接调用 LLM")
            history_context = "\n".join(
                f"用户：{item['question']}\n助手：{item['answer']}" for item in history
            )
            return {"history": history_context, "question": query}, category, []

        
        logger.info("查询为专业咨询, 执行 RAG 流程")
        strategy = self.strategy_selector.select_strategy(query)
        chunks = self.retrieve(query, source_filter=source_filter, strategy=strategy)

        if chunks:
            context, _citations = build_citation_context(chunks)
            logger.info(f"构建上下文完成, 包含 {len(chunks)} 个文档块")
        else:
            context = ""
            logger.info("未检索到相关文档, 上下文为空")

        return self._build_rag_inputs(query, context, history), category, chunks

    def _generation_chain(self, category: str, streaming: bool = False):
        """
        函数功能: 按问题类别和输出方式选择通用或 RAG 的 LCEL 生成链.
        :param category: 分类器返回的“通用知识”或专业问题类别
        :param streaming: False 选择同步链; True 选择流式链
        :return: 已装配完成、可调用 invoke() 或 stream() 的 LCEL Runnable
        """

        if category == "通用知识":
            return self.general_stream_chain if streaming else self.general_chain
        return self.rag_stream_chain if streaming else self.rag_chain

    
    def answer_with_details(self, query: str, source_filter: str = None, history=None) -> dict:
        """
        函数功能: 运行一次非流式完整问答, 校验引用并返回评测所需的详细结果.
        :param query: 用户原始问题
        :param source_filter: 可选知识来源过滤条件
        :param history: 可选会话历史列表
        :return: 包含 answer、category、chunks、citations 和 latency_seconds 的字典

        说明: 准备上下文和生成答案放在同一次调用中, 避免评测器重复执行昂贵检索.
        """
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 来源过滤: {source_filter}")

        chain_inputs, category, chunks = self._prepare(
            query, source_filter=source_filter, history=history
        )
        answer = self._generation_chain(category).invoke(chain_inputs)
        if not answer:
            answer = f"抱歉，处理您的问题时出错。请联系人工客服：{config.CUSTOMER_SERVICE_PHONE}"

        elapsed = time.time() - start_time
        logger.info(f"查询处理完成 (耗时: {elapsed:.2f}s, 查询: '{query}')")
        _context, citations = build_citation_context(chunks)
        answer, used_citations = validate_answer_citations(answer, citations)
        return {
            "answer": answer,
            "category": category,
            "chunks": chunks,
            "citations": [citation.to_dict() for citation in used_citations],
            "latency_seconds": elapsed,
        }

    def answer(self, query: str, source_filter: str = None, history=None) -> str:
        """
        函数功能: 运行非流式问答并只返回最终答案, 保持旧接口兼容.
        :param query: 用户原始问题
        :param source_filter: 可选知识来源过滤条件
        :param history: 可选会话历史列表
        :return: 引用校验后的完整答案文本
        """
        return self.answer_with_details(
            query, source_filter=source_filter, history=history
        )["answer"]

    
    def stream_answer(self, query: str, source_filter: str = None, history=None, details_callback=None) -> Iterator[str]:
        """
        函数功能: 流式生成答案, 逐 token 产出(供前端打字机效果使用).
        :param query: 用户原始问题
        :param source_filter: 可选知识来源过滤条件
        :param history: 可选会话历史列表
        :param details_callback: 生成结束后接收分类和有效引用信息的可选回调
        :return: 生成器, 逐段产出文本
        """
        start_time = time.time()
        logger.info(f"开始处理查询(流式): '{query}', 来源过滤: {source_filter}")

        
        chain_inputs, category, chunks = self._prepare(
            query, source_filter=source_filter, history=history
        )
        answer_parts = []
        for token in self._generation_chain(category, streaming=True).stream(chain_inputs):
            answer_parts.append(token)
            yield token

        # 流式生成结束后，只报告答案中实际引用的来源，避免把“检索到但没引用”的文件显示给用户。
        if details_callback:
            _, citations = build_citation_context(chunks)
            _, used_citations = validate_answer_citations("".join(answer_parts), citations)
            details_callback({"category": category, "citations": [item.to_dict() for item in used_citations]})

        logger.info(f"查询处理完成 (耗时: {time.time() - start_time:.2f}s, 查询: '{query}')")

    
    def generate_answer(self, query: str, source_filter: str = None, history=None, stream: bool = False, details_callback=None):
        """
        函数功能: 生成答案的统一入口.
        :param query: 用户原始问题
        :param source_filter: 可选知识来源过滤条件
        :param history: 可选会话历史列表
        :param stream: False(默认) 返回字符串; True 返回生成器
        :param details_callback: 流式结束后接收分类和引用信息的可选回调
        :return: str 或 Iterator[str]
        """
        if stream:
            return self.stream_answer(query, source_filter=source_filter, history=history, details_callback=details_callback)
        return self.answer(query, source_filter=source_filter, history=history)

if __name__ == '__main__':
    from rag_qa.retriever import HybridRetriever
    from rag_qa.generation.llm import LLMClient

    
    retriever = HybridRetriever()
    llm = LLMClient()

    
    rag_system = RAGSystem(retriever, llm)

    
    print(rag_system.generate_answer('计算圆周率近似值的方法有哪些？', source_filter='ai'))

    
    for token in rag_system.generate_answer('AI学科的课程大纲内容有什么', source_filter='ai', stream=True):
        print(token, end="", flush=True)
