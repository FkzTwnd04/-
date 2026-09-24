"""
调度文档上传任务，并将耗时的解析、切分和向量写入移出 HTTP 请求处理过程。

上传原件按租户目录持久保存，任务状态保存在当前服务进程中，文档处理状态同时写入
MySQL 台账。单机多进程部署时应将内存任务队列换成 Redis/Celery 等共享队列。
"""

import asyncio
from pathlib import Path
import re
from threading import RLock
import uuid
from typing import Any, Dict, Optional

from base.config import config
from base.logger import logger
from rag_qa.corpus.indexer import DocumentIndexer


class IngestService:
    """管理当前进程内的文档任务，并通过信号量限制同时运行的摄取数量。"""

    def __init__(
        self,
        indexer: DocumentIndexer,  # 负责把一份文件从原始字节完整写入向量库的单文档处理器。
        upload_dir: Optional[str] = None,  # 上传原件的持久化目录；缺省读取项目配置。
        max_concurrent: Optional[int] = None,  # 同一进程允许并行执行的索引任务数，防止模型显存被占满。
        max_upload_bytes: Optional[int] = None,  # 单个文件可接受的最大字节数，避免无界上传占满磁盘和内存。
    ):
        self.indexer = indexer
        self.upload_dir = Path(upload_dir or config.DOCUMENT_UPLOAD_DIR).resolve()
        self.max_concurrent = max(1, max_concurrent or config.INGEST_MAX_CONCURRENT)
        self.max_upload_bytes = max_upload_bytes or config.DOCUMENT_MAX_UPLOAD_BYTES
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._tasks: Dict[str, Dict[str, Any]] = {}  # task_id 对应的进度信息，供状态 API 查询。
        self._lock = RLock()  # 保护任务状态字典，避免后台线程与请求线程同时修改记录。

    @staticmethod
    def _safe_filename(filename: str) -> str:
        """移除路径部分和非法字符，防止上传文件名逃出租户目录。"""

        basename = Path(filename or "").name
        safe_name = re.sub(r"[^\w.() -]+", "_", basename, flags=re.UNICODE).strip(" .")
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("文件名不合法")
        return safe_name[:180]

    def submit(
        self,
        raw_bytes: bytes,  # HTTP 上传读取到的完整文件字节；服务校验大小后持久化并交给索引器。
        filename: str,  # 上传文件原名；清理后用于磁盘展示、类型识别和文档标题。
        tenant_id: str = "default",  # 文件归属租户；当前 API 身份鉴权未实现，默认保持单租户。
        source: str = "uploaded",  # 写入 Chunk 的知识来源标签，可用于 RAG 查询过滤。
        source_path: Optional[str] = None,  # 用于判断重复或更新的稳定路径键；默认按租户与文件名生成。
    ) -> str:
        """保存上传文件并安排后台索引任务，立即返回任务编号。"""

        if not isinstance(raw_bytes, bytes) or not raw_bytes:
            raise ValueError("上传文件内容不能为空")
        if len(raw_bytes) > self.max_upload_bytes:
            raise ValueError(f"文件大小超过限制：最大 {self.max_upload_bytes} 字节")

        safe_name = self._safe_filename(filename)
        tenant = self._safe_filename(tenant_id or "default")
        task_id = uuid.uuid4().hex
        destination_dir = self.upload_dir / tenant
        destination_dir.mkdir(parents=True, exist_ok=True)
        stored_path = destination_dir / f"{task_id}_{safe_name}"
        stored_path.write_bytes(raw_bytes)
        stable_source = source_path or f"{tenant}/{safe_name}"

        task = {
            "task_id": task_id,
            "tenant_id": tenant,
            "filename": safe_name,
            "stored_path": str(stored_path),
            "source_path": stable_source,
            "source": source,
            "status": "pending",
            "chunk_count": 0,
            "doc_id": None,
            "error": None,
        }
        with self._lock:
            self._tasks[task_id] = task
        asyncio.get_running_loop().create_task(self._execute(task_id, raw_bytes))
        return task_id

    async def _execute(self, task_id: str, raw_bytes: bytes) -> None:
        """在并发上限内执行同步摄取工作，并把成功或失败写入任务状态。"""

        async with self._semaphore:
            with self._lock:
                task = self._tasks[task_id]
                task["status"] = "indexing"
            try:
                result = await asyncio.to_thread(
                    self.indexer.index_document,
                    raw_bytes,
                    task["filename"],
                    task["tenant_id"],
                    task["source"],
                    task["source_path"],
                    task["stored_path"],
                )
                with self._lock:
                    task.update(
                        {
                            "status": result["status"],
                            "doc_id": result["doc_id"],
                            "chunk_count": result["chunk_count"],
                            "kb_version": result.get("kb_version"),
                        }
                    )
            except Exception as error:
                logger.exception("文档后台摄取失败：task_id=%s", task_id)
                with self._lock:
                    task["status"] = "failed"
                    task["error"] = f"{type(error).__name__}: {error}"

    def status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """返回当前进程仍保留的任务状态；未知任务返回 None。"""

        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task else None

    def list_documents(self, tenant_id: str = "default", limit: int = 100):
        """从 MySQL 台账读取持久文档状态，供文档管理页面展示。"""

        from storage.documents import DocumentStore

        store = DocumentStore(self.indexer.mysql, tenant_id=tenant_id)
        store.init_table()
        return store.list_by_tenant(limit=limit)

    async def retry_failed(self, tenant_id: str = "default") -> Dict[str, Any]:
        """从台账找出失败文档，读取持久原件并重新提交到受并发限制的任务队列。"""

        from storage.documents import DocumentStore

        document_store = DocumentStore(self.indexer.mysql, tenant_id=tenant_id)
        document_store.init_table()
        retried_tasks = []
        skipped_documents = []
        for doc_id in document_store.list_unfinished():
            record = document_store.get(doc_id)
            if not record or record["status"] != "failed" or not record.get("file_path"):
                continue
            original_path = Path(record["file_path"])
            if not original_path.is_file():
                skipped_documents.append({"doc_id": doc_id, "reason": "找不到上传原件"})
                continue
            task_id = self.submit(
                raw_bytes=original_path.read_bytes(),
                filename=original_path.name.split("_", 1)[1],
                tenant_id=tenant_id,
                source_path=record["source_path"],
            )
            retried_tasks.append(task_id)
        return {"task_ids": retried_tasks, "skipped": skipped_documents}
