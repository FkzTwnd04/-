"""
根据策略将原始问句转换成适合检索的查询集合，例如直接查询、HyDE、子查询和回溯改写。
输出会交给 retriever 搜索，多个查询的结果由上层流程汇总去重。
模型改写失败时应保留使用原问题的安全退路，并评估额外查询带来的成本。
"""

from typing import List

from base.logger import logger
from rag_qa.generation.prompts import RAGPrompts

class QueryRewriter:
    """查询改写器: 依赖注入大模型, 自身不持有任何模型客户端."""

    def __init__(self, llm):
        """
        函数功能: 统一模型接口并预先装配 HyDE、子查询和回溯三条 LCEL 改写链.
        :param llm: 大模型调用对象(LLMClient 或 callable), 由外部注入
        :return: None
        """
        from rag_qa.generation.llm import as_llm, build_text_chain

        self.llm = as_llm(llm)
        self.hyde_chain = build_text_chain(RAGPrompts.hyde_prompt(), self.llm)
        self.subquery_chain = build_text_chain(RAGPrompts.subquery_prompt(), self.llm)
        self.backtrack_chain = build_text_chain(RAGPrompts.backtracking_prompt(), self.llm)

    
    def rewrite(self, query: str, mode: str = "direct") -> List[str]:
        """
        函数功能: 按指定模式改写查询.
        :param query: 原始查询
        :param mode: 改写模式 -> direct(不改写) / hyde / subquery / backtrack
        :return: 改写后的查询列表(字符串列表)
                 说明: 返回列表而不是字符串, 是因为 subquery 会产生多个查询; 其余模式返回长度为1的列表.
        """
        if mode == "hyde":
            return [self._hyde(query)]
        if mode == "subquery":
            return self._subqueries(query)
        if mode == "backtrack":
            return [self._backtrack(query)]
        return [query]

    
    def _hyde(self, query: str) -> str:
        """
        函数功能: 通过 HyDE 链生成一段假设答案作为检索查询, 空结果时退回原问题.
        :param query: 用户原始问题
        :return: 假设答案文本; 生成失败时返回原问题
        """
        hypo_answer = self.hyde_chain.invoke({"query": query}).strip()
        
        if not hypo_answer:
            logger.warning("HyDE 生成假设答案失败, 降级为原查询")
            return query
        logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
        return hypo_answer

    
    def _subqueries(self, query: str) -> List[str]:
        """
        函数功能: 通过子查询链拆分复杂问题并清理模型输出中的空行.
        :param query: 用户原始问题
        :return: 子查询列表; 未生成有效内容时返回只包含原问题的列表
        """
        text = self.subquery_chain.invoke({"query": query}).strip()
        subqueries = [q.strip() for q in text.split("\n") if q.strip()]
        if not subqueries:
            logger.warning("未能生成有效的子查询, 降级为原查询")
            return [query]
        logger.info(f"生成的子查询: {subqueries}")
        return subqueries

    
    def _backtrack(self, query: str) -> str:
        """
        函数功能: 通过回溯链把复杂问题简化为更基础的检索问题.
        :param query: 用户原始问题
        :return: 简化后的问题; 生成失败时返回原问题
        """
        simplified = self.backtrack_chain.invoke({"query": query}).strip()
        if not simplified:
            logger.warning("回溯问题生成失败, 降级为原查询")
            return query
        logger.info(f"生成的回溯问题: '{simplified}'")
        return simplified
