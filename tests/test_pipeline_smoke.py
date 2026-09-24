"""
用假模块或替身对象屏蔽部分模型、向量库和存储重依赖，对项目导入及问答流程中的关键分支做快速冒烟检
查。
它能发现接口不匹配、过滤条件遗漏和流程异常等基础问题，但不会验证真实 Milvus 召回质量、外
部服务稳定性或意图阈值效果；
这些需要集成测试和离线评测补充。
"""

import sys
import types

ROOT = r"D:\Desktop\RAGAgent\ra-g-agent\integrated_qa_system"
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
from rag_qa.query.rewriter import QueryRewriter
from rag_qa.router.strategy import STRATEGY_SUBQUERY, StrategySelector
from rag_qa.pipeline.qa import RAGSystem
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
        if "子查询" in prompt:
            return "什么是A\n什么是B"
        return "这是模型的一次性回答"

    def stream(self, prompt, temperature=0.1):
        for ch in "流式回答内容":
            yield ch

class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def hybrid_search_with_rerank(self, query, k=5, source_filter=None):
        self.calls.append({"query": query, "k": k, "source_filter": source_filter})
        return [
            Document(page_content=f"块-{query}-1", metadata={"source": source_filter or "", "timestamp": "2026-01-01"}),
            Document(page_content="重复的块", metadata={"source": source_filter or "", "timestamp": "2026-01-01"}),
        ]

class FakeClassifier:
    def predict_category(self, query):
        return "专业咨询"

class FakeSelector:
    def select_strategy(self, query):
        return STRATEGY_SUBQUERY

vs = FakeVectorStore()
rag = RAGSystem(
    vector_store=vs,
    llm=FakeLLM(),
    classifier=FakeClassifier(),
    strategy_selector=FakeSelector(),
    rewriter=QueryRewriter(FakeLLM()),
)

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
        return [0.1, 0.2], {1: 0.5}

class FakeStore:
    def __init__(self):
        self.last_filters = None

    def hybrid_search_raw(self, dense_vector, sparse_vector, k, filters, dense_weight, sparse_weight):
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
