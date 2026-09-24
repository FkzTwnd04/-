"""
保存文档摄取台账，记录租户、来源、内容指纹、处理状态、文本块数量和失败原因。

该存储层只负责可靠地读写状态，不负责解析文件或操作 Milvus；摄取编排由
rag_qa.corpus.indexer 调用这里的方法完成。
"""

import hashlib
import uuid
from typing import Any, Dict, List, Optional, Tuple

from base.logger import logger


def content_hash(content: Any) -> str:
    """返回内容的 SHA-256 指纹；文本统一换行，二进制文件按原始字节计算。"""

    if isinstance(content, str):
        normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        payload = normalized.encode("utf-8")
    elif isinstance(content, bytes):
        payload = content
    else:
        raise TypeError("content 必须是 str 或 bytes")
    return hashlib.sha256(payload).hexdigest()


class DocumentStore:
    """通过 MySQL 持久化文档状态，所有读写都限定在一个 tenant_id 内。"""

    def __init__(
        self,
        mysql_client,  # 已连接的 MySQL 客户端，用于执行台账 SQL 和提交事务。
        tenant_id: str = "default",  # 当前操作的数据租户；服务鉴权接通前使用默认单租户。
    ):
        self.mysql = mysql_client
        self.tenant_id = tenant_id or "default"

    def init_table(self) -> None:
        """创建文档台账及常用查询索引；IF NOT EXISTS 保留已有记录。"""

        self.mysql.cursor.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                doc_id VARCHAR(64) PRIMARY KEY,
                tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                source_path VARCHAR(1024) NOT NULL,
                file_path VARCHAR(1024) NULL,
                title VARCHAR(512) NOT NULL,
                content_hash CHAR(64) NOT NULL,
                status VARCHAR(24) NOT NULL DEFAULT 'pending',
                chunk_count INT NOT NULL DEFAULT 0,
                retry_count INT NOT NULL DEFAULT 0,
                error_msg TEXT NULL,
                kb_version BIGINT NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uq_documents_tenant_path (tenant_id, source_path(191)),
                KEY idx_documents_tenant_hash (tenant_id, content_hash),
                KEY idx_documents_tenant_status (tenant_id, status)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"""
        )
        self.mysql.connection.commit()

    def need_ingest(self, source_path: str, content: Any) -> Tuple[bool, str]:
        """判断内容是否重复；返回是否需要处理和可复用或新建的文档编号。"""

        fingerprint = content_hash(content)
        tenant = self.tenant_id
        self.mysql.cursor.execute(
            "SELECT doc_id FROM documents WHERE tenant_id = %s AND content_hash = %s "
            "AND status = 'ingested' LIMIT 1",
            (tenant, fingerprint),
        )
        same_content = self.mysql.cursor.fetchone()
        if same_content:
            return False, same_content[0]

        self.mysql.cursor.execute(
            "SELECT doc_id, content_hash, status FROM documents WHERE tenant_id = %s AND source_path = %s LIMIT 1",
            (tenant, source_path),
        )
        same_path = self.mysql.cursor.fetchone()
        if same_path:
            is_ingested_copy = same_path[1] == fingerprint and same_path[2] == "ingested"
            return not is_ingested_copy, same_path[0]
        return True, uuid.uuid4().hex

    def classify_content(self, source_path: str, content: Any) -> Tuple[str, Optional[str]]:
        """区分新文档、相同内容重复提交和同路径内容更新。"""

        fingerprint = content_hash(content)
        self.mysql.cursor.execute(
            "SELECT doc_id FROM documents WHERE tenant_id = %s AND content_hash = %s "
            "AND status = 'ingested' LIMIT 1",
            (self.tenant_id, fingerprint),
        )
        duplicate = self.mysql.cursor.fetchone()
        if duplicate:
            return "duplicate", duplicate[0]

        self.mysql.cursor.execute(
            "SELECT doc_id, content_hash, status FROM documents WHERE tenant_id = %s AND source_path = %s LIMIT 1",
            (self.tenant_id, source_path),
        )
        same_path = self.mysql.cursor.fetchone()
        if same_path:
            state = "duplicate" if same_path[1] == fingerprint and same_path[2] == "ingested" else (
                "retry" if same_path[1] == fingerprint else "updated"
            )
            return state, same_path[0]
        return "new", None

    def upsert_pending(self, source_path: str, title: str, content: Any, file_path: Optional[str] = None) -> str:
        """把文件登记或更新为 pending，确保开始解析前已有可追踪的文档编号。"""

        fingerprint = content_hash(content)
        tenant = self.tenant_id
        self.mysql.cursor.execute(
            "SELECT doc_id FROM documents WHERE tenant_id = %s AND source_path = %s LIMIT 1",
            (tenant, source_path),
        )
        existing = self.mysql.cursor.fetchone()
        document_id = existing[0] if existing else uuid.uuid4().hex
        self.mysql.cursor.execute(
            """INSERT INTO documents
                (doc_id, tenant_id, source_path, file_path, title, content_hash, status, chunk_count, error_msg)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending', 0, NULL)
                ON DUPLICATE KEY UPDATE file_path = VALUES(file_path), title = VALUES(title), content_hash = VALUES(content_hash),
                    status = 'pending', chunk_count = 0, error_msg = NULL""",
            (document_id, tenant, source_path, file_path, title, fingerprint),
        )
        self.mysql.connection.commit()
        return document_id

    def mark_ingested(self, doc_id: str, chunk_count: int, kb_version: str) -> None:
        """记录向量写入成功、文本块数量和完成时的知识版本。"""

        self.mysql.cursor.execute(
            "UPDATE documents SET status = 'ingested', chunk_count = %s, kb_version = %s, error_msg = NULL "
            "WHERE doc_id = %s AND tenant_id = %s",
            (chunk_count, int(kb_version), doc_id, self.tenant_id),
        )
        self.mysql.connection.commit()

    def mark_failed(self, doc_id: str, error: str) -> None:
        """记录失败状态和截断后的错误文本，供重试或排查使用。"""

        self.mysql.cursor.execute(
            "UPDATE documents SET status = 'failed', retry_count = retry_count + 1, error_msg = %s "
            "WHERE doc_id = %s AND tenant_id = %s",
            (str(error)[:4000], doc_id, self.tenant_id),
        )
        self.mysql.connection.commit()
        logger.error("文档摄取失败：tenant=%s, doc_id=%s, error=%s", self.tenant_id, doc_id, error)

    def list_unfinished(self) -> List[str]:
        """列出当前租户需要恢复或重试的文档编号。"""

        self.mysql.cursor.execute(
            "SELECT doc_id FROM documents WHERE tenant_id = %s AND status IN ('pending', 'failed') ORDER BY updated_at",
            (self.tenant_id,),
        )
        return [row[0] for row in self.mysql.cursor.fetchall()]

    def mark_stale(self, active_paths: List[str]) -> int:
        """将不在本次有效文件清单中的已入库文档标记为 stale，并返回影响行数。"""

        if not active_paths:
            self.mysql.cursor.execute(
                "UPDATE documents SET status = 'stale' WHERE tenant_id = %s AND status = 'ingested'",
                (self.tenant_id,),
            )
        else:
            placeholders = ", ".join(["%s"] * len(active_paths))
            self.mysql.cursor.execute(
                f"UPDATE documents SET status = 'stale' WHERE tenant_id = %s AND status = 'ingested' "
                f"AND source_path NOT IN ({placeholders})",
                [self.tenant_id, *active_paths],
            )
        changed = self.mysql.cursor.rowcount
        self.mysql.connection.commit()
        return changed

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """按租户和文档编号读取台账；不存在或属于其他租户时返回 None。"""

        self.mysql.cursor.execute(
            "SELECT doc_id, tenant_id, source_path, file_path, title, content_hash, status, chunk_count, "
            "retry_count, error_msg, kb_version, created_at, updated_at "
            "FROM documents WHERE doc_id = %s AND tenant_id = %s",
            (doc_id, self.tenant_id),
        )
        row = self.mysql.cursor.fetchone()
        if not row:
            return None
        columns = (
            "doc_id", "tenant_id", "source_path", "file_path", "title", "content_hash", "status",
            "chunk_count", "retry_count", "error_msg", "kb_version", "created_at", "updated_at",
        )
        return dict(zip(columns, row))

    def list_by_tenant(self, limit: int = 100) -> List[Dict[str, Any]]:
        """按更新时间倒序列出租户文档，限制返回数量避免一次加载整张台账。"""

        safe_limit = max(1, min(int(limit), 500))
        self.mysql.cursor.execute(
            "SELECT doc_id, source_path, title, status, chunk_count, retry_count, error_msg, updated_at "
            f"FROM documents WHERE tenant_id = %s ORDER BY updated_at DESC LIMIT {safe_limit}",
            (self.tenant_id,),
        )
        rows = self.mysql.cursor.fetchall()
        columns = ("doc_id", "source_path", "title", "status", "chunk_count", "retry_count", "error_msg", "updated_at")
        return [dict(zip(columns, row)) for row in rows]
