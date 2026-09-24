"""
执行单份文档的幂等摄取：登记台账、判重、解析切分、写入向量库并记录结果。

该模块按租户隔离文档台账和向量删除条件；上传文件原件由调用方持久化，发生失败时可重新提交。
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

from base.logger import logger
from rag_qa.corpus.dedup import DocumentDeduplicator
from rag_qa.corpus.processor import process_documents
from rag_qa.schema.chunk import to_chunk
from storage.documents import DocumentStore


class DocumentIndexer:
    """将单个上传文件从原始字节送到当前租户的 Milvus 集合。"""

    def __init__(
        self,
        mysql_client,  # 用于建立摄取台账和知识版本表的 MySQL 客户端。
        vector_store,  # 提供 add(chunks) 与 delete_document(doc_id, tenant_id) 的检索存储。
        processor=process_documents,  # 可替换的目录解析切分函数；正式运行默认使用项目处理器。
    ):
        self.mysql = mysql_client
        self.vector_store = vector_store
        self.processor = processor

    def _bump_kb_version(self, tenant_id: str) -> int:
        """递增租户知识库版本，令使用旧知识版本构造的缓存键不再命中。"""

        self.mysql.cursor.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_base_versions (
                tenant_id VARCHAR(64) PRIMARY KEY,
                version BIGINT NOT NULL DEFAULT 1,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"""
        )
        self.mysql.cursor.execute(
            "INSERT INTO knowledge_base_versions (tenant_id, version) VALUES (%s, 2) "
            "ON DUPLICATE KEY UPDATE version = version + 1",
            (tenant_id,),
        )
        self.mysql.cursor.execute(
            "SELECT version FROM knowledge_base_versions WHERE tenant_id = %s",
            (tenant_id,),
        )
        version = int(self.mysql.cursor.fetchone()[0])
        self.mysql.connection.commit()
        return version

    def index_document(
        self,
        raw_bytes: bytes,  # 上传文件的原始字节；以此计算幂等指纹，避免解析器改动影响判重。
        filename: str,  # 用户上传的文件名；用于识别文件类型和在台账中展示。
        tenant_id: str = "default",  # 文件归属租户；必须来自可信调用方，当前 API 暂使用默认租户。
        source: str = "uploaded",  # 知识来源分类；写进 Chunk 供查询过滤和页面展示。
        source_path: Optional[str] = None,  # 稳定文件来源键；省略时由租户和文件名生成。
        stored_file_path: Optional[str] = None,  # 上传原件持久路径；用于失败后重新读取并重试。
    ) -> Dict[str, Any]:
        """完成单文档摄取；完全重复的内容返回 duplicate 并跳过切分和向量写入。"""

        if not isinstance(raw_bytes, bytes) or not raw_bytes:
            raise ValueError("上传文件必须包含非空二进制内容")
        safe_filename = Path(filename).name
        if not safe_filename:
            raise ValueError("文件名不能为空")

        tenant = tenant_id or "default"
        stable_source_path = source_path or f"{tenant}/{safe_filename}"
        document_store = DocumentStore(self.mysql, tenant_id=tenant)
        document_store.init_table()
        deduplicator = DocumentDeduplicator(document_store)
        state, existing_id = deduplicator.classify(raw_bytes, stable_source_path)
        if state == "duplicate" and existing_id:
            return {"doc_id": existing_id, "status": "duplicate", "chunk_count": 0}

        title = Path(safe_filename).stem
        doc_id = document_store.upsert_pending(stable_source_path, title, raw_bytes, file_path=stored_file_path)
        if state == "updated" and existing_id:
            self.vector_store.delete_document(existing_id, tenant_id=tenant)

        try:
            with TemporaryDirectory(prefix="rag_ingest_") as temporary_directory:
                temporary_file = Path(temporary_directory) / safe_filename
                temporary_file.write_bytes(raw_bytes)
                chunks_as_documents = self.processor(temporary_directory)

            chunks = []
            for index, item in enumerate(chunks_as_documents):
                chunk = to_chunk(item)
                chunk.doc_id = doc_id
                chunk.tenant_id = tenant
                chunk.source = source
                chunk.filename = safe_filename
                original_chunk_id = chunk.chunk_id or str(index)
                chunk.chunk_id = f"{doc_id}:{original_chunk_id}"
                if chunk.parent_id:
                    chunk.parent_id = f"{doc_id}:{chunk.parent_id}"
                chunks.append(chunk)

            if not chunks:
                raise ValueError("文档解析后没有得到可入库的文本块，请检查文件格式和内容")

            written_count = self.vector_store.add(chunks)
            if written_count != len(chunks):
                raise RuntimeError(f"向量库写入数量不一致：计划 {len(chunks)}，实际 {written_count}")
            version = self._bump_kb_version(tenant)
            document_store.mark_ingested(doc_id, written_count, version)
            logger.info("文档摄取完成：tenant=%s, doc_id=%s, chunks=%s", tenant, doc_id, written_count)
            return {
                "doc_id": doc_id,
                "status": "ingested",
                "chunk_count": written_count,
                "kb_version": version,
            }
        except Exception as error:
            document_store.mark_failed(doc_id, str(error))
            raise
