"""
提供文档上传、摄取状态查询、租户文档列表和失败任务重试接口。

路由只做 HTTP 文件接收与响应组织；解析、切分、哈希判重和 Milvus 写入委托给 IngestService。
当前服务还没有用户身份鉴权，因此暂时只使用默认单租户，不接受客户端自报 tenant_id。
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.deps import get_ingest_service
from base.config import config

router = APIRouter()


@router.post("/api/docs/upload", status_code=202)
async def upload_document(
    file: UploadFile = File(...),  # 用户上传的原始文件；由 FastAPI 分块暂存，读取后交给后台服务。
    source: str = Form("ai"),  # 文档知识来源标签，必须属于配置中的有效来源，供 RAG 检索过滤。
):
    """接收文件并排入后台摄取，立即返回 task_id 供客户端查询进度。"""

    if source not in config.VALID_SOURCES:
        raise HTTPException(status_code=400, detail=f"不支持的文档来源：{source}")

    chunks = []
    total_bytes = 0
    while True:
        part = await file.read(1024 * 1024)
        if not part:
            break
        total_bytes += len(part)
        if total_bytes > config.DOCUMENT_MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="上传文件超过大小限制")
        chunks.append(part)
    if not total_bytes:
        raise HTTPException(status_code=400, detail="上传文件不能为空")

    raw_bytes = b"".join(chunks)
    try:
        task_id = get_ingest_service().submit(raw_bytes, file.filename or "", source=source)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"task_id": task_id, "status": "pending", "filename": file.filename, "size_bytes": total_bytes}


@router.get("/api/docs/tasks/{task_id}")
async def get_ingest_status(task_id: str):
    """读取当前进程中的上传任务进度；任务服务重启后可改查文档台账列表。"""

    task = get_ingest_service().status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或已不在当前进程的任务缓存中")
    return task


@router.get("/api/docs")
async def list_documents(limit: int = 100):
    """列出默认租户的持久文档台账，limit 最大为 500。"""

    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit 必须在 1 到 500 之间")
    return {"documents": get_ingest_service().list_documents(limit=limit)}


@router.post("/api/docs/retry")
async def retry_failed_documents():
    """重新提交台账中仍有原件可读取的失败文档。"""

    return await get_ingest_service().retry_failed()
