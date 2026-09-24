"""保存在线请求的运营记录，并维护从用户反馈到 FAQ 更新的人工审核闭环。

请求日志用于统计各答案层的流量和耗时，反馈用于观察用户对答案的评价，候选表用于整理
重复出现且落到 L3 的问题。当前问题归并只忽略标点和空白差异，不会自动生成答案或做语义聚类。
"""

from datetime import datetime, timedelta
import hashlib
import re
from typing import Dict, List, Optional

from base.logger import logger


class OpsStore:
    """将问答日志、用户反馈和 FAQ 候选保存在 MySQL 中。"""

    def __init__(self, mysql_client, tenant_id: str = "default"):
        """mysql_client 是项目共享的 MySQL 连接；tenant_id 限定本实例可读写的数据范围。"""
        self.mysql = mysql_client
        self.tenant_id = tenant_id or "default"
        self.init_tables()

    def init_tables(self) -> None:
        """创建运营闭环所需的三张表；重复启动不会覆盖已有数据。"""
        statements = (
            """CREATE TABLE IF NOT EXISTS request_logs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                tenant_id VARCHAR(64) NOT NULL,
                query_hash CHAR(64) NOT NULL,
                query_text TEXT NOT NULL,
                answer_text MEDIUMTEXT,
                hit_layer VARCHAR(16) NOT NULL,
                intent VARCHAR(128),
                latency_ms INT NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                KEY idx_request_tenant_time (tenant_id, created_at),
                KEY idx_request_layer (hit_layer)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci""",
            """CREATE TABLE IF NOT EXISTS request_feedback (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                request_id BIGINT NOT NULL,
                tenant_id VARCHAR(64) NOT NULL,
                rating TINYINT NOT NULL,
                comment TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uq_feedback_request (request_id, tenant_id),
                KEY idx_feedback_tenant_time (tenant_id, created_at)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci""",
            """CREATE TABLE IF NOT EXISTS faq_candidates (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                tenant_id VARCHAR(64) NOT NULL,
                cluster_key CHAR(64) NOT NULL,
                cluster_question VARCHAR(2000) NOT NULL,
                sample_queries JSON NOT NULL,
                frequency INT NOT NULL DEFAULT 1,
                status VARCHAR(16) NOT NULL DEFAULT 'pending',
                reviewed_answer MEDIUMTEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uq_candidate_cluster (tenant_id, cluster_key),
                KEY idx_candidate_status (tenant_id, status, frequency)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci""",
        )
        try:
            for statement in statements:
                self.mysql.cursor.execute(statement)
            self.mysql.connection.commit()
        except Exception:
            self.mysql.connection.rollback()
            raise

    @staticmethod
    def _redact(text: str) -> str:
        """遮盖常见联系方式和证件号，降低运营日志保存个人信息的风险。"""
        text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号]", text)
        text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[邮箱]", text)
        return re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[证件号]", text)

    @staticmethod
    def _normalize(query: str) -> str:
        """统一大小写、空白和标点，供简单的相似问句聚合使用。"""
        return re.sub(r"[\W_]+", "", query.casefold(), flags=re.UNICODE)

    def log_request(
        self,
        query: str,
        answer: str = "",
        hit_layer: str = "L3",
        intent: Optional[str] = None,
        latency_ms: int = 0,
    ) -> Optional[int]:
        """写入一次请求并返回日志编号；数据库异常只记告警，不中断在线回答。"""
        safe_query = self._redact(str(query or ""))[:10000]
        safe_answer = self._redact(str(answer or ""))[:50000]
        digest = hashlib.sha256(self._normalize(safe_query).encode("utf-8")).hexdigest()
        try:
            self.mysql.cursor.execute(
                "INSERT INTO request_logs "
                "(tenant_id, query_hash, query_text, answer_text, hit_layer, intent, latency_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (self.tenant_id, digest, safe_query, safe_answer, hit_layer, intent, max(0, int(latency_ms))),
            )
            request_id = int(self.mysql.cursor.lastrowid)
            self.mysql.connection.commit()
            return request_id
        except Exception as error:
            self.mysql.connection.rollback()
            logger.warning("保存问答运营日志失败，不影响回答：%s", error)
            return None

    def record_feedback(self, request_id: int, rating: int, comment: str = "") -> None:
        """保存用户评价；rating 只接受 1（有帮助）或 -1（没帮助），重复提交会更新。"""
        if rating not in (-1, 1):
            raise ValueError("rating 只能是 1 或 -1")
        self.mysql.cursor.execute(
            "SELECT id FROM request_logs WHERE id = %s AND tenant_id = %s",
            (request_id, self.tenant_id),
        )
        if not self.mysql.cursor.fetchone():
            raise ValueError("找不到该租户下的问答记录")
        self.mysql.cursor.execute(
            "INSERT INTO request_feedback (request_id, tenant_id, rating, comment) VALUES (%s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE rating = VALUES(rating), comment = VALUES(comment)",
            (request_id, self.tenant_id, rating, self._redact(comment or "")[:4000]),
        )
        self.mysql.connection.commit()

    def find_gaps(
        self, min_freq: int = 3, days: int = 7, hit_layers: Optional[List[str]] = None
    ) -> List[Dict]:
        """汇总时间窗内重复出现的未解决问句；先用归一化完全匹配，避免引入不可靠的伪聚类。"""
        if min_freq < 1 or days < 1:
            raise ValueError("min_freq 和 days 必须是正整数")
        layers = hit_layers or ["L3"]
        if not layers:
            return []
        placeholders = ", ".join(["%s"] * len(layers))
        since = datetime.now() - timedelta(days=days)
        self.mysql.cursor.execute(
            f"SELECT query_text, hit_layer FROM request_logs "
            f"WHERE tenant_id = %s AND created_at >= %s AND hit_layer IN ({placeholders})",
            (self.tenant_id, since, *layers),
        )
        groups = {}
        for query_text, hit_layer in self.mysql.cursor.fetchall():
            normalized = self._normalize(query_text or "")
            if not normalized:
                continue
            key = (normalized, hit_layer)
            group = groups.setdefault(key, {"queries": [], "frequency": 0})
            group["frequency"] += 1
            if len(group["queries"]) < 10 and query_text not in group["queries"]:
                group["queries"].append(query_text)
        results = []
        for (_normalized, hit_layer), group in groups.items():
            if group["frequency"] < min_freq:
                continue
            results.append({
                "cluster_question": group["queries"][0],
                "sample_queries": group["queries"],
                "frequency": group["frequency"],
                "hit_layer": hit_layer,
            })
        return sorted(results, key=lambda item: item["frequency"], reverse=True)

    def add_candidate(self, cluster_question: str, sample_queries: List[str], freq: int) -> int:
        """登记待审核 FAQ；相同规范化问题更新频次和样例，不重复创建候选。"""
        question = (cluster_question or "").strip()
        if not question or freq < 1:
            raise ValueError("候选问题不能为空，freq 必须大于 0")
        normalized = self._normalize(question)
        key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        safe_samples = [self._redact(str(item))[:2000] for item in sample_queries if str(item).strip()][:100]
        safe_samples = list(dict.fromkeys(safe_samples or [self._redact(question)]))
        import json

        self.mysql.cursor.execute(
            "INSERT INTO faq_candidates (tenant_id, cluster_key, cluster_question, sample_queries, frequency) "
            "VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE "
            "frequency = GREATEST(frequency, VALUES(frequency)), sample_queries = VALUES(sample_queries), "
            "status = IF(status = 'rejected', 'pending', status)",
            (self.tenant_id, key, self._redact(question)[:2000], json.dumps(safe_samples, ensure_ascii=False), freq),
        )
        self.mysql.connection.commit()
        self.mysql.cursor.execute(
            "SELECT id FROM faq_candidates WHERE tenant_id = %s AND cluster_key = %s", (self.tenant_id, key)
        )
        return int(self.mysql.cursor.fetchone()[0])

    def list_candidates(self, status: str = "pending", limit: int = 100) -> List[Dict]:
        """读取指定状态的候选；limit 限制单次管理查询返回的最大条数。"""
        allowed = {"pending", "approved", "rejected"}
        if status not in allowed:
            raise ValueError("status 必须是 pending、approved 或 rejected")
        limit = max(1, min(int(limit), 500))
        self.mysql.cursor.execute(
            "SELECT id, cluster_question, sample_queries, frequency, status, created_at "
            "FROM faq_candidates WHERE tenant_id = %s AND status = %s "
            "ORDER BY frequency DESC, id DESC LIMIT %s",
            (self.tenant_id, status, limit),
        )
        import json

        return [
            {
                "id": int(row[0]), "cluster_question": row[1],
                "sample_queries": json.loads(row[2]) if isinstance(row[2], str) else row[2],
                "frequency": int(row[3]), "status": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
            }
            for row in self.mysql.cursor.fetchall()
        ]

    def review_candidate(self, candidate_id: int, approved: bool, answer: str = "") -> None:
        """拒绝候选只更新状态；通过时将人工答案写入 FAQ 并递增知识版本使索引缓存失效。"""
        self.mysql.cursor.execute(
            "SELECT cluster_question, status FROM faq_candidates WHERE id = %s AND tenant_id = %s FOR UPDATE",
            (candidate_id, self.tenant_id),
        )
        row = self.mysql.cursor.fetchone()
        if not row:
            raise ValueError("找不到待审核候选")
        if approved and not (answer or "").strip():
            raise ValueError("审核通过时必须提供人工确认的标准答案")
        try:
            status = "approved" if approved else "rejected"
            if approved:
                self.mysql.cursor.execute(
                    "INSERT INTO faq_entries (tenant_id, question, answer, is_active) VALUES (%s, %s, %s, 1)",
                    (self.tenant_id, row[0], answer.strip()),
                )
                self.mysql.cursor.execute(
                    "INSERT INTO knowledge_base_versions (tenant_id, version) VALUES (%s, 2) "
                    "ON DUPLICATE KEY UPDATE version = version + 1",
                    (self.tenant_id,),
                )
            self.mysql.cursor.execute(
                "UPDATE faq_candidates SET status = %s, reviewed_answer = %s WHERE id = %s AND tenant_id = %s",
                (status, answer.strip() if approved else None, candidate_id, self.tenant_id),
            )
            self.mysql.connection.commit()
        except Exception:
            self.mysql.connection.rollback()
            raise

    def reuse_stats(self, days: int = 7) -> Dict[str, object]:
        """统计各命中层请求量、平均耗时和点踩率，时间窗由 days 指定。"""
        if days < 1:
            raise ValueError("days 必须是正整数")
        since = datetime.now() - timedelta(days=days)
        self.mysql.cursor.execute(
            "SELECT r.hit_layer, COUNT(*), AVG(r.latency_ms), "
            "SUM(CASE WHEN f.rating = -1 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN f.rating IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM request_logs r LEFT JOIN request_feedback f "
            "ON f.request_id = r.id AND f.tenant_id = r.tenant_id "
            "WHERE r.tenant_id = %s AND r.created_at >= %s GROUP BY r.hit_layer",
            (self.tenant_id, since),
        )
        rows = self.mysql.cursor.fetchall()
        total = sum(int(row[1]) for row in rows)
        return {
            str(row[0]): {
                "requests": int(row[1]),
                "share": (int(row[1]) / total) if total else 0.0,
                "average_latency_ms": float(row[2] or 0),
                "thumbs_down": int(row[3] or 0),
                "feedback_count": int(row[4] or 0),
                "thumbs_down_rate": (int(row[3] or 0) / int(row[4])) if row[4] else 0.0,
            }
            for row in rows
        }

    def list_requests(self, limit: int = 50) -> List[Dict[str, object]]:
        """列出最近问答运营日志；query_text/answer_text 在入库前已做常见个人信息脱敏。"""
        safe_limit = max(1, min(int(limit), 200))
        self.mysql.cursor.execute(
            "SELECT r.id, r.query_text, r.answer_text, r.hit_layer, r.intent, r.latency_ms, "
            "r.created_at, f.rating FROM request_logs r LEFT JOIN request_feedback f "
            "ON f.request_id = r.id AND f.tenant_id = r.tenant_id "
            "WHERE r.tenant_id = %s ORDER BY r.created_at DESC, r.id DESC LIMIT %s",
            (self.tenant_id, safe_limit),
        )
        return [
            {
                "request_id": int(row[0]),
                "query": row[1],
                "answer": (row[2] or "")[:500],
                "hit_layer": row[3],
                "intent": row[4],
                "latency_ms": int(row[5] or 0),
                "created_at": row[6].isoformat() if row[6] else None,
                "rating": row[7],
            }
            for row in self.mysql.cursor.fetchall()
        ]
