"""
实现 FAQ 的 BM25 字面匹配、向量语义兜底、租户知识版本和 Redis 精确答案缓存。

只有查询通过个性化/时效问题过滤，并达到 BM25 或向量检索的分数与 top1-top2 差值门槛时，
才会复用人工维护的 FAQ 答案。它负责 FAQ 答案匹配，不等同于意图代表问句检索。
"""

import hashlib
import re
from threading import RLock
from typing import List, Optional, Tuple, Union

from rank_bm25 import BM25Okapi

from base.config import config
from base.logger import logger
from storage.preprocess import preprocess_text

DENY_CACHE_PATTERNS: List[str] = [
    "余额", "账单", "订单号", "手机号", "身份证", "验证码", "密码",
    "我的", "折扣", "报价", "价格",
]

class BM25Search:
    """FAQ 字面匹配(L2a): 原始 BM25 分 + 绝对阈值 + 区分度双判据, 命中即直接复用答案."""

    def __init__(self, redis_client, mysql_client, tenant_id: str = ""):
        """
        :param redis_client: Redis 客户端, 存 FAQ 索引与命中缓存
        :param mysql_client: MySQL 客户端, 取 FAQ 标准问答对
        :param tenant_id: 租户标识 —— FAQ 索引与缓存都必须按租户隔离, 缺省表示单租户部署
        """
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.tenant_id = tenant_id

        
        self.bm25: Optional[BM25Okapi] = None      
        self.raw_questions: List[str] = []         
        self.faq_rows = []
        self.kb_version: str = ""                  

        
        self.threshold = config.FAQ_L2A_THRESHOLD
        self.margin = config.FAQ_L2A_MARGIN
        self.vector_threshold = config.FAQ_L2B_THRESHOLD
        self.vector_margin = config.FAQ_L2B_MARGIN
        self._faq_vectors = None  # FAQ 问题向量只在 BM25 无可信命中时生成并缓存在当前进程。
        self._lock = RLock()  # 保护本实例的 BM25 索引重建，避免并发请求同时写入索引状态。
        self._tables_ready = False

        
        self._load_index()

    
    def _load_index(self) -> None:
        """
        按 tenant_id 读取有效 FAQ，并在 MySQL 中的知识版本变化后重建当前进程的 BM25 索引。
        索引只缓存问题文本和答案记录；答案更新通过递增版本号触发重载。
        """
        self._ensure_tables()
        version = self._get_kb_version()
        if self.kb_version and self.kb_version == version:
            return

        with self._lock:
            version = self._get_kb_version()
            if self.kb_version and self.kb_version == version:
                return
            self.mysql_client.cursor.execute(
                "SELECT id, question, answer FROM faq_entries "
                "WHERE tenant_id = %s AND is_active = 1 ORDER BY id ASC",
                (self.tenant_id or "default",),
            )
            rows = self.mysql_client.cursor.fetchall()
            self.faq_rows = [
                {"id": row[0], "question": row[1], "answer": row[2]}
                for row in rows
                if row[1] and preprocess_text(row[1])
            ]
            self.raw_questions = [row["question"] for row in self.faq_rows]
            tokenized_questions = [preprocess_text(question) for question in self.raw_questions]
            self.bm25 = BM25Okapi(tokenized_questions) if tokenized_questions else None
            self._faq_vectors = None
            self.kb_version = version
            logger.info("FAQ BM25 索引已加载：tenant=%s, count=%s, version=%s", self.tenant_id, len(self.faq_rows), version)

    def _ensure_tables(self) -> None:
        """创建 FAQ 答案表与租户知识版本表；两张表均可重复初始化。"""

        if self._tables_ready:
            return
        statements = (
            """CREATE TABLE IF NOT EXISTS faq_entries (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                question VARCHAR(2000) NOT NULL,
                answer TEXT NOT NULL,
                is_active TINYINT(1) NOT NULL DEFAULT 1,
                hit_count BIGINT NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                KEY idx_faq_tenant_active (tenant_id, is_active)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci""",
            """CREATE TABLE IF NOT EXISTS knowledge_base_versions (
                tenant_id VARCHAR(64) PRIMARY KEY,
                version BIGINT NOT NULL DEFAULT 1,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci""",
        )
        for statement in statements:
            self.mysql_client.cursor.execute(statement)
        self.mysql_client.connection.commit()
        self._tables_ready = True

    def _get_kb_version(self) -> str:
        """读取当前租户知识库版本；首次使用时创建从 1 开始的版本记录。"""

        tenant = self.tenant_id or "default"
        self.mysql_client.cursor.execute(
            "SELECT version FROM knowledge_base_versions WHERE tenant_id = %s",
            (tenant,),
        )
        row = self.mysql_client.cursor.fetchone()
        if row:
            return str(row[0])
        self.mysql_client.cursor.execute(
            "INSERT IGNORE INTO knowledge_base_versions (tenant_id, version) VALUES (%s, 1)",
            (tenant,),
        )
        self.mysql_client.connection.commit()
        return "1"

    
    def search(
        self, query: str, threshold: float = None, include_metadata: bool = False
    ) -> Union[Tuple[Optional[str], bool], Tuple[Optional[str], bool, Optional[str]]]:
        """
        按 L1 精确缓存、L2a BM25、L2b 向量相似度顺序查找 FAQ；
        L2a/L2b 都必须同时通过绝对分数阈值和 top1-top2 margin，未命中时返回 need_rag=True。
        搜索前会拒绝个性化和动态问题，命中后返回知识库中的人工维护答案，不从模型生成答案。
        :param include_metadata: True 时在原有二元组后增加命中层，供运营记录区分 L1、L2a 和 L2b。
        :return: 默认返回 (答案, 是否需要继续走 RAG)；include_metadata=True 时多返回命中层。
        """
        def result(answer, need_rag, hit_layer=None):
            return (answer, need_rag, hit_layer) if include_metadata else (answer, need_rag)

        if not query or not isinstance(query, str):
            logger.error("无效查询: 查询为空或者非字符串类型")
            return result(None, True)
        threshold = self.threshold if threshold is None else threshold
        if not self.is_cacheable(query):
            return result(None, True)

        self._load_index()
        tenant = self.tenant_id or "default"
        normalized_query = re.sub(r"\s+", " ", query.strip().casefold())
        cache_digest = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()
        cache_key = f"faq:l1:{tenant}:{self.kb_version}:{cache_digest}"
        try:
            cached_answer = self.redis_client.client.get(cache_key)
            if cached_answer:
                self.record_hit(question=query, faq_id=0, hit_layer="L1")
                return result(cached_answer, False, "L1")
        except Exception as error:
            logger.warning("FAQ 精确缓存读取失败，继续 BM25：%s", error)

        matched = None
        hit_layer = None
        if self.bm25 is not None:
            scores = [float(score) for score in self.bm25.get_scores(preprocess_text(query))]
            ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            top_index = ranked[0]
            top_score = scores[top_index]
            second_score = scores[ranked[1]] if len(ranked) > 1 else 0.0
            if top_score >= threshold and top_score - second_score >= self.margin:
                matched = self.faq_rows[top_index]
                hit_layer = "L2a"

        if matched is None:
            try:
                matched = self._semantic_match(query)
            except Exception as error:
                logger.warning("FAQ 向量兜底不可用，继续进入 RAG：%s", error)
                matched = None
            if matched is not None:
                hit_layer = "L2b"
        if matched is None:
            return result(None, True)

        answer = matched["answer"]
        if not answer:
            return result(None, True)
        try:
            if self.is_cacheable(query):
                self.redis_client.client.setex(cache_key, config.FAQ_CACHE_TTL_SECONDS, answer)
        except Exception as error:
            logger.warning("FAQ 精确答案缓存写入失败：%s", error)
        self.record_hit(question=query, faq_id=matched["id"], hit_layer=hit_layer)
        return result(answer, False, hit_layer)

    def _semantic_match(self, query: str):
        """以向量余弦和 top1-top2 margin 作为 FAQ L2b 兜底；向量模型按需加载。"""
        if not self.faq_rows:
            return None
        from rag_qa.retriever.embedding import BGEM3Embedder

        with self._lock:
            if self._faq_vectors is None:
                embedder = BGEM3Embedder()
                vectors = embedder.embed(self.raw_questions)["dense"]
                self._faq_vectors = [list(map(float, vector)) for vector in vectors]
                self._faq_embedder = embedder
            faq_vectors = self._faq_vectors
            embedder = self._faq_embedder
        query_vector = list(map(float, embedder.embed([query])["dense"][0]))

        def cosine(left, right):
            dot = sum(a * b for a, b in zip(left, right))
            left_norm = sum(value * value for value in left) ** 0.5
            right_norm = sum(value * value for value in right) ** 0.5
            return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

        scores = [cosine(query_vector, vector) for vector in faq_vectors]
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        top_index = ranked[0]
        top_score = scores[top_index]
        second_score = scores[ranked[1]] if len(ranked) > 1 else 0.0
        if top_score < self.vector_threshold or top_score - second_score < self.vector_margin:
            return None
        logger.info(
            "FAQ 向量命中：tenant=%s, score=%.4f, margin=%.4f",
            self.tenant_id, top_score, top_score - second_score,
        )
        return self.faq_rows[top_index]

    
    def invalidate(self) -> None:
        """
        做什么: 主动让当前索引失效(下次查询会重建).
        为什么这么做: 运营审核通过新 FAQ、或修改/下线某条 FAQ 后, 必须能被立刻感知.
                      靠 TTL 自然过期存在时间窗, 这段时间里用户会拿到旧答案(甚至是已下线的答案).
        实现要点: 由管理接口(新增/修改 FAQ)调用, 或直接 bump kb_version 让缓存 key 集体失效.
        """
        self._ensure_tables()
        tenant = self.tenant_id or "default"
        self.mysql_client.cursor.execute(
            "INSERT INTO knowledge_base_versions (tenant_id, version) VALUES (%s, 2) "
            "ON DUPLICATE KEY UPDATE version = version + 1",
            (tenant,),
        )
        self.mysql_client.connection.commit()
        with self._lock:
            self.bm25 = None
            self.faq_rows = []
            self.raw_questions = []
            self.kb_version = ""
            self._faq_vectors = None

    
    @staticmethod
    def is_cacheable(text: str) -> bool:
        """
        做什么: 判断一条问答能不能进缓存.
        为什么这么做: 缓存是"一次写入、多次返回", 中间没有二次校验,
                      所以个性化/时效性内容必须在写入前就拦掉, 而不是读出后再补救.
        实现要点: 按 DENY_CACHE_PATTERNS 做关键词匹配, 命中返回 False(不缓存).
        """
        if not isinstance(text, str) or not text.strip():
            return False
        normalized = text.casefold()
        return not any(term.casefold() in normalized for term in DENY_CACHE_PATTERNS)

    
    def record_hit(self, question: str, faq_id: int, hit_layer: str) -> None:
        """
        做什么: 记录一次命中到 MySQL, 供运营闭环使用.
        为什么这么做: 没有统计就没有闭环 —— 你无法回答"这一层到底拦住了多少流量、省了多少钱",
                      而这两个数字是成本治理唯一的依据(storage/ops.py 会汇总它们).
        实现要点: 只累加 hit_count 或写明细表; 不要在大流量路径上同步写库(先写 Redis 再异步落库).
        """
        if hit_layer not in {"L1", "L2a", "L2b"}:
            raise ValueError("FAQ 命中层只能是 L1、L2a 或 L2b")
        if faq_id:
            self.mysql_client.cursor.execute(
                "UPDATE faq_entries SET hit_count = hit_count + 1 "
                "WHERE tenant_id = %s AND id = %s",
                (self.tenant_id or "default", faq_id),
            )
            self.mysql_client.connection.commit()
        logger.info("FAQ 命中：tenant=%s, layer=%s, question=%s", self.tenant_id, hit_layer, question)

    def upsert_faq(self, question: str, answer: str, faq_id: Optional[int] = None) -> int:
        """新增或修改 FAQ 答案并递增知识库版本，使旧索引和缓存失效。"""

        if not question.strip() or not answer.strip():
            raise ValueError("FAQ 问题和答案都不能为空")
        self._ensure_tables()
        tenant = self.tenant_id or "default"
        if faq_id is None:
            self.mysql_client.cursor.execute(
                "INSERT INTO faq_entries (tenant_id, question, answer) VALUES (%s, %s, %s)",
                (tenant, question.strip(), answer.strip()),
            )
            faq_id = self.mysql_client.cursor.lastrowid
        else:
            self.mysql_client.cursor.execute(
                "UPDATE faq_entries SET question = %s, answer = %s, is_active = 1 "
                "WHERE tenant_id = %s AND id = %s",
                (question.strip(), answer.strip(), tenant, faq_id),
            )
        self.mysql_client.connection.commit()
        self.invalidate()
        return int(faq_id)

if __name__ == '__main__':
    from storage.cache import RedisClient
    from storage.db import MySQLClient

    redis_client = RedisClient()
    mysql_client = MySQLClient()
    faq = BM25Search(redis_client, mysql_client)
    print(faq.search(query='Python好学吗'))
