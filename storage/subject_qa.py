"""检索项目已有的 jpkb 学科问答数据。

这个表保存学科名称、问题和答案，是用户已有的题库数据，不是 FAQ 候选表或意图样本表。
本模块按固定时间间隔把题库加载到进程内：完整问题精确命中时直接复用答案；近似问法使用
BM25 分数饱和归一化到 0—1，只有达到 0.85 才复用答案，未命中时交给 Jev 决定后续路线。
"""

import re
import time
from threading import RLock
from typing import Any, Dict, List, Optional

import jieba
from rank_bm25 import BM25Okapi

from base.config import config
from base.logger import logger


class SubjectQASearch:
    """在现有 jpkb 表里查学科问答，不要求把题库复制到 faq_entries。"""

    def __init__(
        self,
        mysql_client,  # 已连接到项目数据库的客户端；查询表为 jpkb。
        refresh_seconds: int = None,  # 题库索引缓存秒数；到期后重新从 MySQL 读一次更新。
        threshold: Optional[float] = None,  # 归一化 BM25 分数下限；只有严格大于 0.85 才复用近似答案。
        margin: Optional[float] = None,  # 兼容旧调用参数；当前按用户要求只使用 0.85 分数门槛。
    ):
        """
        函数功能: 保存 MySQL 客户端, 确定索引刷新间隔(至少 1 秒)和归一化分数门槛, 并初始化空的题库缓存与 BM25 索引.
        :param mysql_client: 已连接的 MySQL 客户端
        :param refresh_seconds: 索引缓存秒数, 为空或 0 时使用 config.HOT_QUESTION_REFRESH_SECONDS
        :param threshold: 归一化 BM25 分数门槛, 为 None 时使用 config.HOT_QUESTION_BM25_THRESHOLD
        :param margin: 兼容旧调用的参数, 当前未使用
        :return: None
        """
        self.mysql = mysql_client
        self.refresh_seconds = max(1, refresh_seconds or config.HOT_QUESTION_REFRESH_SECONDS)
        self.threshold = config.HOT_QUESTION_BM25_THRESHOLD if threshold is None else threshold
        self._lock = RLock()
        self._loaded_at = 0.0
        self._rows: List[Dict[str, Any]] = []
        self._exact_questions: Dict[str, Dict[str, Any]] = {}
        self._tokens: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    @staticmethod
    def _normalize(question: str) -> str:
        """忽略大小写、空格和标点，让用户输入与题库原题的小格式差异不影响精确匹配。"""

        return re.sub(r"[\W_]+", "", question.casefold(), flags=re.UNICODE)

    @staticmethod
    def _tokenize(question: str) -> List[str]:
        """使用 jieba 对题目分词；建索引和查询时必须使用同样的分词规则。"""

        return [word.strip() for word in jieba.lcut(question.casefold()) if word.strip()]

    def _refresh_index(self) -> None:
        """缓存过期时读取题库并重建精确匹配字典和 BM25 索引。"""

        now = time.monotonic()
        if now - self._loaded_at < self.refresh_seconds:
            return

        with self._lock:
            now = time.monotonic()
            if now - self._loaded_at < self.refresh_seconds:
                return
            try:
                self.mysql.cursor.execute(
                    "SELECT id, subject_name, question, answer FROM jpkb "
                    "WHERE question IS NOT NULL AND answer IS NOT NULL ORDER BY id ASC"
                )
                rows = self.mysql.cursor.fetchall()
            except Exception as error:
                self.mysql.connection.rollback()
                self._rows = []
                self._exact_questions = {}
                self._tokens = []
                self._bm25 = None
                self._loaded_at = now - max(self.refresh_seconds - 5, 0)
                logger.warning("读取已有题库 jpkb 失败，稍后重试；请检查数据库名和表结构：%s", error)
                return

            self._rows = [
                {
                    "id": row[0],
                    "subject": row[1] or "",
                    "question": str(row[2]).strip(),
                    "answer": str(row[3]).strip(),
                }
                for row in rows
                if row[2] and row[3] and str(row[2]).strip() and str(row[3]).strip()
            ]
            self._exact_questions = {}
            for item in self._rows:
                normalized = self._normalize(item["question"])
                if normalized:
                    self._exact_questions.setdefault(normalized, item)
            self._tokens = [self._tokenize(item["question"]) for item in self._rows]
            self._bm25 = BM25Okapi(self._tokens) if self._tokens else None
            self._loaded_at = now
            logger.info("已有题库 jpkb 索引已加载：有效问答=%s", len(self._rows))

    def search(self, question: str) -> Optional[Dict[str, Any]]:
        """原题直接命中；近似题将 BM25 分数归一化后只按配置门槛决定是否返回答案。"""

        if not isinstance(question, str) or not question.strip():
            return None
        self._refresh_index()
        normalized = self._normalize(question)
        exact = self._exact_questions.get(normalized)
        if exact:
            logger.info("题库 jpkb 精确命中：id=%s, subject=%s", exact["id"], exact["subject"])
            return {**exact, "hit_layer": "jpkb_exact", "score": 1.0, "raw_score": None, "margin": None}

        if self._bm25 is None:
            logger.info("题库 jpkb 暂无可检索问答，记录数=%s", len(self._rows))
            return None

        query_tokens = self._tokenize(question)
        if not query_tokens:
            return None
        scores = [float(score) for score in self._bm25.get_scores(query_tokens)]
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        top_index = ranked[0]
        top_score = scores[top_index]
        second_score = scores[ranked[1]] if len(ranked) > 1 else 0.0
        score_margin = top_score - second_score
        normalized_score = top_score / (top_score + 1.0) if top_score > 0 else 0.0
        if normalized_score <= self.threshold:
            logger.info(
                "题库 jpkb 未命中：raw_score=%.4f, normalized_score=%.4f, threshold=%.4f",
                top_score, normalized_score, self.threshold,
            )
            return None

        matched = self._rows[top_index]
        logger.info(
            "题库 jpkb BM25 命中：id=%s, subject=%s, raw_score=%.4f, normalized_score=%.4f",
            matched["id"], matched["subject"], top_score, normalized_score,
        )
        return {**matched, "hit_layer": "jpkb_bm25", "score": normalized_score, "raw_score": top_score, "margin": score_margin}
