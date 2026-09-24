"""用 BM25 检查用户问题是否命中人工维护的意图代表问句。

BM25 只返回通过 0.85 分数线的代表问句匹配结果，不生成答案，也不使用向量兜底；
RAG/通用问题的业务路由由 Jev 分类器负责。
"""

from threading import RLock
import time
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from base.config import config
from base.logger import logger
from storage.preprocess import preprocess_text

class IntentRetriever:
    """按租户缓存代表问句索引，并仅用 BM25 分数阈值返回匹配记录。"""

    def __init__(
        self,
        sample_store,  # 提供 list_active(tenant_id) 的存储对象，负责返回人工维护的代表问句。
        bm25_threshold: Optional[float] = None,  # BM25 top1 原始分数下限；低于它时不接受词面命中。
        index_ttl_seconds: Optional[int] = None,  # 内存索引允许使用的秒数；到期后重新读取最新样本。
    ):
        self.sample_store = sample_store
        self.bm25_threshold = bm25_threshold if bm25_threshold is not None else config.INTENT_BM25_THRESHOLD
        self.index_ttl_seconds = index_ttl_seconds or config.INTENT_INDEX_TTL_SECONDS
        self._indexes: Dict[str, Dict[str, Any]] = {}  # 每个租户各自缓存 BM25 索引，避免不同租户的样本混用。
        self._lock = RLock()  # 索引刷新时加锁，避免并发请求同时重建同一租户索引。

    def invalidate(self, tenant_id: str = "default") -> None:
        """删除一个租户的进程内索引；新增、修改或停用代表问句后调用。"""

        with self._lock:
            self._indexes.pop(tenant_id or "default", None)

    def _get_index(self, tenant_id: str) -> Dict[str, Any]:
        """返回尚未过期的索引；第一次访问或缓存过期时从存储重建。"""

        now = time.monotonic()
        with self._lock:
            cached = self._indexes.get(tenant_id)
            if cached and now - cached["loaded_at"] < self.index_ttl_seconds:
                return cached

            samples = self.sample_store.list_active(tenant_id)
            usable_samples = [
                sample
                for sample in samples
                if sample.get("query", "").strip() and sample.get("intent", "").strip()
            ]
            tokenized_questions = [preprocess_text(sample["query"]) for sample in usable_samples]
            index = {
                "loaded_at": now,
                "samples": usable_samples,
                "bm25": BM25Okapi(tokenized_questions) if tokenized_questions else None,
                "tokenized_questions": tokenized_questions,
            }
            self._indexes[tenant_id] = index
            return index

    @staticmethod
    def _rank(scores: List[float]) -> List[int]:
        """按分数从高到低返回候选下标；下标对应索引中的代表问句。"""

        return sorted(range(len(scores)), key=lambda position: scores[position], reverse=True)

    def _match_record(
        self,
        sample: Dict[str, Any],  # 被检索命中的代表问句记录，含有标签及可选固定答案。
        layer: str,  # 产生命中的检索阶段；当前只会是 BM25。
        score: float,  # 命中候选的 BM25 原始分，供日志与评测分析。
        margin: float,  # 第一候选与第二候选的分差，记录在结果中便于排查阈值。
    ) -> Dict[str, Any]:
        """整理 BM25 命中的代表问句、标签和分数，供日志及评测记录使用。"""

        return {
            "sample_id": sample["sample_id"],  # 代表问句的稳定编号，用于回查或维护样本。
            "representative_query": sample["query"],  # 实际命中的历史问法，用于解释分类结果。
            "intent": sample["intent"],  # 命中样本对应的意图标签。
            "layer": layer,  # 标记本次代表问句由 BM25 命中。
            "score": float(score),  # 保存 top1 绝对分数，支持日志分析和阈值调优。
            "margin": float(margin),  # 保存 top1-top2 差值，支持判断候选是否足够明确。
        }

    def match(
        self,
        query: str,  # 用户当前输入的问题；作为 BM25 查询文本与意图代表问句比较。
        tenant_id: str = "default",  # 代表问句所属租户；调用方必须传可信身份，不能直接信任客户端字段。
    ) -> Optional[Dict[str, Any]]:
        """只按 BM25 原始分数判断是否达到阈值；未命中时返回 None。"""

        if not isinstance(query, str) or not query.strip():
            return None
        tenant = tenant_id or "default"
        index = self._get_index(tenant)
        samples = index["samples"]
        if not samples:
            logger.info("意图代表问句检索未运行：intent_samples 中没有启用的样本，请先导入代表问句 JSONL")
            return None

        query_tokens = preprocess_text(query)
        bm25_index = index["bm25"]
        if bm25_index is not None and query_tokens:
            bm25_scores = [float(score) for score in bm25_index.get_scores(query_tokens)]
            ranked = self._rank(bm25_scores)
            top_score = bm25_scores[ranked[0]]
            second_score = bm25_scores[ranked[1]] if len(ranked) > 1 else 0.0
            score_margin = top_score - second_score
            if top_score >= self.bm25_threshold:
                return self._match_record(samples[ranked[0]], "bm25", top_score, score_margin)
        logger.info("意图代表问句未命中 BM25 阈值：threshold=%.4f", self.bm25_threshold)
        return None
