"""
用假模块或替身对象屏蔽部分模型、向量库和存储重依赖，对项目导入及问答流程中的关键分支做快速冒烟检
查。
它能发现接口不匹配、过滤条件遗漏和流程异常等基础问题，但不会验证真实 Milvus 召回质量、外
部服务稳定性或意图阈值效果；
这些需要集成测试和离线评测补充。
"""

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def install_stub(name, attrs=(), submodule_of=None):
    """若真实包可导入就用真实的, 否则注册一个占位模块."""
    try:
        __import__(name)
        return False
    except Exception:
        pass
    mod = types.ModuleType(name)
    for a in attrs:
        setattr(mod, a, type(a, (object,), {}))
    sys.modules[name] = mod
    if submodule_of and submodule_of in sys.modules:
        setattr(sys.modules[submodule_of], name.split(".")[-1], mod)
    return True

stubbed = []
for name, attrs, parent in [
    ("torch", [], None),
    ("torch.cuda", [], "torch"),
    ("numpy", ["argmax"], None),
    ("milvus_model", [], None),
    ("milvus_model.hybrid", ["BGEM3EmbeddingFunction"], "milvus_model"),
    ("pymilvus", ["MilvusClient", "DataType", "AnnSearchRequest", "WeightedRanker"], None),
    ("sentence_transformers", ["CrossEncoder"], None),
    ("transformers", ["BertTokenizer", "BertForSequenceClassification", "Trainer", "TrainingArguments"], None),
    ("sklearn", [], None),
    ("sklearn.model_selection", ["train_test_split"], "sklearn"),
    ("sklearn.metrics", ["classification_report", "confusion_matrix"], "sklearn"),
]:
    if install_stub(name, attrs, parent):
        stubbed.append(name)
print(f"[INFO] 被桩件替换的模块: {stubbed or '无(真实依赖已安装)'}")

try:
    from langchain_core.documents import Document
except Exception:
    class Document:
        def __init__(self, page_content="", metadata=None):
            """
            函数功能: langchain_core 缺失时的最小 Document 实现, 保存正文和元数据(metadata 为空时用空字典).
            :param page_content: 文档正文
            :param metadata: 元数据字典
            :return: None
            """
            self.page_content = page_content
            self.metadata = metadata or {}

    lc = types.ModuleType("langchain_core")
    lcd = types.ModuleType("langchain_core.documents")
    lcd.Document = Document
    lc.documents = lcd
    sys.modules["langchain_core"] = lc
    sys.modules["langchain_core.documents"] = lcd
    print("[INFO] langchain_core 缺失, 使用最小实现")

from rag_qa.schema.chunk import Chunk
from rag_qa.generation.prompts import RAGPrompts
from rag_qa.generation.llm import as_runnable
from rag_qa.query.rewriter import QueryRewriter
from rag_qa.router.strategy import STRATEGY_SUBQUERY, StrategySelector
from rag_qa.pipeline.qa import RAGSystem
from langchain_core.output_parsers import StrOutputParser
print("[OK] 新分层包导入成功")

doc = Document(page_content="正文内容", metadata={"source": "ai", "timestamp": "2026-09-22",
                                                 "parent_id": "p1", "parent_content": "父块正文", "page": 3})
chunk = Chunk.from_document(doc)
assert chunk.text == "正文内容" and chunk.source == "ai" and chunk.page == 3
back = chunk.to_document()
assert back.page_content == "正文内容" and back.metadata["source"] == "ai"
print("[OK] Chunk <-> Document 双向转换正确")

assert StrategySelector._normalize("假设问题检索（HyDE）") == "假设问题检索"
assert StrategySelector._normalize("子查询检索") == "子查询检索"
assert StrategySelector._normalize("  回溯问题检索  ") == "回溯问题检索"
assert StrategySelector._normalize("乱七八糟的输出") == "直接检索"
print("[OK] 策略名归一化正确(带括号/空格/无效输出都能收敛)")

class FakeLLM:
    def complete(self, prompt, temperature=0.1):
        """
        函数功能: 模拟 LLM 一次性补全: 提示包含“子查询”时返回两行子查询“什么是A\\n什么是B”, 否则返回固定回答.
        :param prompt: 提示词文本
        :param temperature: 采样温度, 替身中不使用
        :return: 固定字符串
        """
        if "子查询" in prompt:
            return "什么是A\n什么是B"
        return "这是模型的一次性回答"

    def stream(self, prompt, temperature=0.1):
        """
        函数功能: 模拟 LLM 流式输出, 忽略提示词, 逐字产出“流式回答内容”.
        :param prompt: 提示词文本, 替身中不使用
        :param temperature: 采样温度, 替身中不使用
        :return: 逐字符产出的生成器
        """
        for ch in "流式回答内容":
            yield ch

class FakeVectorStore:
    def __init__(self):
        """
        函数功能: 初始化向量库替身, 用 calls 列表记录每次检索调用的参数.
        :return: None
        """
        self.calls = []

    def hybrid_search_with_rerank(self, query, k=5, source_filter=None):
        """
        函数功能: 模拟向量库的混合检索+重排, 记录调用参数; 与真实 HybridRetriever 一样返回 Chunk, 另混入一个旧式 Document 检验兼容.
        :param query: 检索查询
        :param k: 返回数量, 仅被记录
        :param source_filter: 来源过滤条件, 被记录并写入返回块的 source
        :return: 一个含查询文本的 Chunk 和一个跨查询重复的 Document
        """
        self.calls.append({"query": query, "k": k, "source_filter": source_filter})
        return [
            Chunk(text=f"块-{query}-1", source=source_filter or "", publish_time="2026-01-01"),
            Document(page_content="重复的块", metadata={"source": source_filter or "", "timestamp": "2026-01-01"}),
        ]

class FakeClassifier:
    def predict_category(self, query):
        """
        函数功能: 模拟意图分类器, 无论输入都返回“专业咨询”.
        :param query: 用户查询, 替身中不使用
        :return: 固定类别字符串“专业咨询”
        """
        return "专业咨询"

hyde_chain = RAGPrompts.hyde_prompt() | as_runnable(FakeLLM()) | StrOutputParser()
assert hyde_chain.invoke({"query": "什么是 RAG"}).strip() == "这是模型的一次性回答"
stream_chain = (
    RAGPrompts.hyde_prompt()
    | as_runnable(FakeLLM(), streaming=True)
    | StrOutputParser()
)
assert "".join(stream_chain.stream({"query": "什么是 RAG"})) == "流式回答内容"
print("[OK] LCEL 文本链支持 invoke 与逐段 stream")

class FakeSelector:
    def select_strategy(self, query):
        """
        函数功能: 模拟检索策略选择器, 固定返回子查询检索策略.
        :param query: 用户查询, 替身中不使用
        :return: STRATEGY_SUBQUERY
        """
        return STRATEGY_SUBQUERY

vs = FakeVectorStore()
rag = RAGSystem(
    vector_store=vs,
    llm=FakeLLM(),
    classifier=FakeClassifier(),
    strategy_selector=FakeSelector(),
    rewriter=QueryRewriter(FakeLLM()),
)
assert hasattr(rag.rag_chain, "invoke") and hasattr(rag.rag_stream_chain, "stream")
assert hasattr(rag.general_chain, "invoke") and hasattr(rag.general_stream_chain, "stream")
assert hasattr(rag.rewriter.hyde_chain, "invoke")
print("[OK] 改写与最终生成步骤均装配为 LCEL 链")

ans = rag.generate_answer("测试问题", source_filter="ai", history=[
    {"question": "上一轮问题", "answer": "上一轮答案"},
    {"question": "无效条目"},
])
assert isinstance(ans, str), f"stream=False 应返回字符串, 实际: {type(ans)}"
print(f"[OK] 非流式返回 str: {ans!r}")

assert len(vs.calls) == 2, f"子查询应触发 2 次检索, 实际: {len(vs.calls)}"
assert all(c["source_filter"] == "ai" for c in vs.calls), f"source_filter 未透传: {vs.calls}"
print(f"[OK] source_filter 已透传到每一次检索, 共 {len(vs.calls)} 次")

streamed = rag.generate_answer("测试问题", source_filter="ai", stream=True)
assert hasattr(streamed, "__iter__") and not isinstance(streamed, str), "stream=True 应返回生成器"
text = "".join(streamed)
assert text == "流式回答内容", f"流式内容不符: {text!r}"
print(f"[OK] 流式返回生成器, 拼接结果: {text!r}")

raw = [
    Chunk(text="A"), Chunk(text="B"), Chunk(text="A"), Chunk(text="C"),
]
deduped = RAGSystem._dedup(raw)
assert len(deduped) == 3, f"去重后应为 3 块, 实际: {len(deduped)}"
print(f"[OK] 去重逻辑正确, 4 块去重后剩 {len(deduped)} 块")

chunks = rag.retrieve("测试问题", source_filter="ai", strategy=STRATEGY_SUBQUERY)
assert all(isinstance(c, Chunk) for c in chunks), "检索结果的每个元素都应是 Chunk"
assert len(chunks) == min(3, rag.candidate_m), f"应被 candidate_m 截取, 实际: {len(chunks)}"
print(f"[OK] 检索返回 {len(chunks)} 块(去重后 3 块, 已按 candidate_m={rag.candidate_m} 截取), 类型统一为 Chunk")

from rag_qa.retriever.milvus_store import build_filter_expr

assert build_filter_expr(None) == "", "无过滤条件时应返回空串"
assert build_filter_expr({"source": "ai"}) == "source == 'ai'"
expr = build_filter_expr({"source": "ai", "tenant_id": "t1"})
assert expr == "source == 'ai' and tenant_id == 't1'", f"多条件拼接错误: {expr}"
assert build_filter_expr({"source": "ai", "tenant_id": ""}) == "source == 'ai'", "空值条件应被跳过"

inj = build_filter_expr({"tenant_id": "t1' or source == 'secret"})

expected = "tenant_id == 't1\\' or source == \\'secret'"
assert inj == expected, f"单引号未正确转义, 存在表达式注入风险: {inj}"
print(f"[OK] 过滤表达式构造正确, 注入已转义: {inj}")

from rag_qa.schema.chunk import to_chunk

assert to_chunk(chunk) is chunk, "传入 Chunk 时应原样返回"
assert isinstance(to_chunk(doc), Chunk), "传入 Document 时应转成 Chunk"
assert to_chunk({"text": "字典块", "source": "ai"}).text == "字典块", "传入字典时应转成 Chunk"
print("[OK] to_chunk 支持 Chunk / Document / dict 三种输入")

from rag_qa.retriever.hybrid import HybridRetriever
from rag_qa.reranker.rerank import Reranker

class FakeEmbedder:
    def embed_one(self, text):
        """
        函数功能: 模拟向量编码器, 不加载模型, 固定返回稠密向量 [0.1, 0.2] 和稀疏向量 {1: 0.5}.
        :param text: 待编码文本, 替身中不使用
        :return: (稠密向量, 稀疏向量) 元组
        """
        return [0.1, 0.2], {1: 0.5}

class FakeStore:
    def __init__(self):
        """
        函数功能: 初始化 Milvus 存储替身, last_filters 用于记录最近一次检索收到的过滤条件.
        :return: None
        """
        self.last_filters = None

    def hybrid_search_raw(self, dense_vector, sparse_vector, k, filters, dense_weight, sparse_weight):
        """
        函数功能: 模拟 Milvus 原始混合检索, 记录过滤条件并返回 3 个固定子块命中(c1、c2 属于父块 p1, c3 属于父块 p2).
        :param dense_vector: 稠密查询向量, 替身中不使用
        :param sparse_vector: 稀疏查询向量, 替身中不使用
        :param k: 返回数量, 替身中不使用
        :param filters: 过滤条件, 保存到 last_filters
        :param dense_weight: 稠密检索权重, 替身中不使用
        :param sparse_weight: 稀疏检索权重, 替身中不使用
        :return: 含 entity 字段的命中字典列表
        """
        self.last_filters = filters
        
        return [
            {"entity": {"id": "c1", "text": "子块1", "parent_id": "p1", "parent_content": "父块1正文",
                        "source": "ai", "timestamp": "2026-01-01", "tenant_id": "t1", "page": 1}},
            {"entity": {"id": "c2", "text": "子块2", "parent_id": "p1", "parent_content": "父块1正文",
                        "source": "ai", "timestamp": "2026-01-01", "tenant_id": "t1", "page": 1}},
            {"entity": {"id": "c3", "text": "子块3", "parent_id": "p2", "parent_content": "父块2正文",
                        "source": "ai", "timestamp": "2026-01-02", "tenant_id": "t1", "page": 5}},
        ]

class FakeReranker:
    """按文本长度打分, 用来验证排序真的生效(不加载真实模型)."""

    def rerank(self, query, chunks, top_k=None):
        """
        函数功能: 模拟重排器, 以文本长度作为 score 写回每个块并按分数降序排序, 给定 top_k 时截取前 top_k 个.
        :param query: 查询文本, 替身中不使用
        :param chunks: 待重排的 Chunk 列表(会被原地写入 score)
        :param top_k: 保留数量, 为空时返回全部
        :return: 排序后的 Chunk 列表
        """
        for c in chunks:
            c.score = float(len(c.text))
        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        return ranked[:top_k] if top_k else ranked

store = FakeStore()
retriever = HybridRetriever(
    store=store, embedder=FakeEmbedder(), reranker=FakeReranker(), candidate_m=2
)
hits = retriever.search("测试问题", k=5, filters={"source": "ai", "tenant_id": "t1"})

assert store.last_filters == {"source": "ai", "tenant_id": "t1"}, "过滤条件未下推到检索引擎"
assert len(hits) == 2, f"3 个子块提升去重后应为 2 个父块, 实际: {len(hits)}"
assert all(isinstance(h, Chunk) for h in hits), "检索结果应统一为 Chunk"
assert {h.chunk_id for h in hits} == {"p1", "p2"}, f"父块ID不正确: {[h.chunk_id for h in hits]}"
assert hits[0].score >= hits[1].score, "重排后未按分数降序"
assert hits[0].page in (1, 5), "页码等溯源字段应保留"
print(f"[OK] 混合检索编排正确: 3 子块 -> {len(hits)} 父块, 过滤条件已下推, 已按分数降序")

old_style = retriever.hybrid_search_with_rerank("测试问题", k=5, source_filter="ai")
assert len(old_style) == 2, "兼容旧签名的检索结果应一致"
print("[OK] 旧方法名 hybrid_search_with_rerank 仍可用")

off = Reranker(enabled=False)
three = [Chunk(text="A"), Chunk(text="BB"), Chunk(text="CCC")]
result_off = off.rerank("q", list(three), top_k=2)
assert [c.text for c in result_off] == ["A", "BB"], "关闭重排时应保持原顺序"
assert Reranker().rerank("q", [Chunk(text="only")], top_k=1)[0].text == "only", "单候选应直接返回"
assert Reranker().rerank("q", []) == [], "空列表应返回空"
print("[OK] 重排开关与边界情况正确(关闭时直通, 单候选/空列表不报错)")

print("\n===== 全部冒烟测试通过 =====")
