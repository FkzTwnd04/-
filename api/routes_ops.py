"""提供在线问答后的反馈采集和知识运营管理入口。

路由调用 storage/ops.py 保存评价、统计命中层，并将高频 L3 问题整理成 FAQ 候选；
只有人工提交确认答案后才会写入 FAQ。该接口当前没有用户身份鉴权，适用于本地单租户环境。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.deps import get_qa_system


router = APIRouter(prefix="/api/ops", tags=["knowledge-operations"])


class FeedbackRequest(BaseModel):
    """request_id 指向问答记录；rating 为 1（有帮助）或 -1（没帮助）。"""
    request_id: int = Field(gt=0)
    rating: int
    comment: Optional[str] = Field(default="", max_length=4000)


class ReviewRequest(BaseModel):
    """approved 为通过/拒绝；通过时 answer 必须是人工确认的标准答案。"""
    approved: bool
    answer: Optional[str] = Field(default="", max_length=50000)


@router.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """把对指定回答的反馈写入运营库，供后续统计误命中率。"""
    try:
        get_qa_system().ops_store.record_feedback(request.request_id, request.rating, request.comment or "")
        return {"status": "ok", "request_id": request.request_id}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="保存反馈失败") from error


@router.get("/stats")
def get_stats(days: int = Query(default=7, ge=1, le=365)):
    """按时间窗口返回各答案层的请求占比、平均耗时和点踩情况。"""
    try:
        return {"days": days, "layers": get_qa_system().ops_store.reuse_stats(days)}
    except Exception as error:
        raise HTTPException(status_code=500, detail="读取运营统计失败") from error


@router.get("/requests")
def list_requests(limit: int = Query(default=50, ge=1, le=200)):
    """返回近期脱敏问答日志，前端运营页面据此查看请求和命中层。"""
    try:
        return {"items": get_qa_system().ops_store.list_requests(limit)}
    except Exception as error:
        raise HTTPException(status_code=500, detail="读取问答日志失败") from error


@router.post("/candidates/build")
def build_candidates(
    min_freq: int = Query(default=3, ge=1, le=100000),
    days: int = Query(default=7, ge=1, le=365),
):
    """把高频 L3 问题生成待审核候选；此步骤不会自动编写或发布答案。"""
    try:
        store = get_qa_system().ops_store
        gaps = store.find_gaps(min_freq=min_freq, days=days)
        candidates = [
            store.add_candidate(item["cluster_question"], item["sample_queries"], item["frequency"])
            for item in gaps
        ]
        return {"created_or_updated": len(candidates), "candidate_ids": candidates}
    except Exception as error:
        raise HTTPException(status_code=500, detail="生成知识候选失败") from error


@router.get("/candidates")
def list_candidates(
    status: str = Query(default="pending"),
    limit: int = Query(default=100, ge=1, le=500),
):
    """读取待审核或已处理的候选，便于运营人员确认要不要补充 FAQ。"""
    try:
        return {"items": get_qa_system().ops_store.list_candidates(status, limit)}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="读取候选失败") from error


@router.post("/candidates/{candidate_id}/review")
def review_candidate(candidate_id: int, request: ReviewRequest):
    """人工通过候选并提交标准答案，或拒绝候选；通过时同步使 FAQ 索引版本失效。"""
    try:
        get_qa_system().ops_store.review_candidate(candidate_id, request.approved, request.answer or "")
        return {"status": "approved" if request.approved else "rejected", "candidate_id": candidate_id}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="审核候选失败") from error
