"""
提供离线评测集上传、后台评测任务状态查询和最新报告读取接口。

评测数据通过上传传入，不允许客户端指定服务端任意文件路径；评测工作在响应后由
FastAPI 后台线程运行，结果仍由 rag_qa.evaluation.runner 保存为 JSON 报告。
"""

import json
from pathlib import Path
import re
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from base.config import config
from rag_qa.evaluation.dataset import load_dataset
from rag_qa.evaluation.runner import RAGEvaluationRunner

router = APIRouter()
EVALUATION_ROOT = Path(config.PROJECT_ROOT) / "rag_qa" / "evaluation"
REPORT_ROOT = EVALUATION_ROOT / "results"
JOB_ROOT = REPORT_ROOT / "jobs"
DATASET_ROOT = EVALUATION_ROOT / "uploaded_datasets"


def _write_job(job_path: Path, job: dict) -> None:
    """持久保存任务状态，避免只靠内存状态导致服务重启后完全不可查。"""

    job_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = job_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(job_path)


def _run_evaluation(job_id: str, dataset_path: str, mode: str, run_name: str) -> None:
    """在后台执行离线评测，并将成功报告路径或错误原因写入任务状态文件。"""

    job_path = JOB_ROOT / f"{job_id}.json"
    try:
        _write_job(job_path, {"job_id": job_id, "status": "running", "mode": mode, "run_name": run_name})
        result = RAGEvaluationRunner().run(
            dataset_path=dataset_path,
            mode=mode,
            output_dir=str(REPORT_ROOT),
            run_name=run_name,
        )
        _write_job(job_path, {"job_id": job_id, "status": "completed", "mode": mode, "result": result})
    except Exception as error:
        _write_job(
            job_path,
            {"job_id": job_id, "status": "failed", "mode": mode, "error": f"{type(error).__name__}: {error}"},
        )


@router.post("/api/eval/runs", status_code=202)
async def start_evaluation(
    background_tasks: BackgroundTasks,  # FastAPI 响应后执行评测函数，避免 HTTP 请求等待整批问题完成。
    file: UploadFile = File(...),  # JSONL 评测数据文件；内容会校验后保存到项目评测数据目录。
    mode: str = Form("rag"),  # 选择 rag 问答评测或 intent 意图评测。
    run_name: str = Form("api_eval"),  # 报告名称前缀；服务端还会追加唯一编号避免覆盖。
):
    """接收一份 JSONL 数据并启动后台 RAG 或意图评测任务。"""

    if mode not in {"rag", "intent"}:
        raise HTTPException(status_code=400, detail="mode 只能是 rag 或 intent")
    if not file.filename or Path(file.filename).suffix.casefold() != ".jsonl":
        raise HTTPException(status_code=400, detail="评测数据文件必须使用 .jsonl 后缀")

    upload_bytes = await file.read()
    if not upload_bytes:
        raise HTTPException(status_code=400, detail="评测集文件不能为空")
    if len(upload_bytes) > config.DOCUMENT_MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="评测集文件超过大小限制")
    try:
        upload_text = upload_bytes.decode("utf-8-sig")
        staged_path = DATASET_ROOT / f"{uuid.uuid4().hex}.jsonl"
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_text(upload_text, encoding="utf-8")
        load_dataset(str(staged_path))  # 提交后台任务前先验证 JSONL 字段、行号和样本编号。
    except (UnicodeDecodeError, ValueError) as error:
        if "staged_path" in locals() and staged_path.exists():
            staged_path.unlink()
        raise HTTPException(status_code=400, detail=f"评测集格式错误：{error}") from error

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name.strip())[:60] or "api_eval"
    job_id = uuid.uuid4().hex
    unique_run_name = f"{safe_name}_{job_id[:8]}"
    _write_job(
        JOB_ROOT / f"{job_id}.json",
        {"job_id": job_id, "status": "pending", "mode": mode, "run_name": unique_run_name},
    )
    background_tasks.add_task(_run_evaluation, job_id, str(staged_path), mode, unique_run_name)
    return {"job_id": job_id, "status": "pending", "mode": mode}


@router.get("/api/eval/runs/{job_id}")
async def get_evaluation_status(job_id: str):
    """读取评测任务状态；job_id 只允许命中任务目录下的 UUID 文件名。"""

    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=400, detail="job_id 格式不正确")
    job_path = JOB_ROOT / f"{job_id}.json"
    if not job_path.is_file():
        raise HTTPException(status_code=404, detail="评测任务不存在")
    return json.loads(job_path.read_text(encoding="utf-8"))


@router.get("/api/eval/latest")
async def get_latest_evaluation():
    """返回最近生成的一份评测报告；无历史报告时返回 404。"""

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = sorted(
        (path for path in REPORT_ROOT.glob("*.json") if path.parent != JOB_ROOT),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        raise HTTPException(status_code=404, detail="还没有已完成的评测报告")
    return json.loads(reports[0].read_text(encoding="utf-8"))
