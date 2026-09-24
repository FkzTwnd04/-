"""
创建 FastAPI 应用并完成中间件、业务路由、静态文件和首页的装配，同时在应用生命周期中触发
配置自检。
应用工厂只负责服务组装，问答决策交由 routes_qa.py 与 pipeline/；
上线时还需按部署域名收紧 CORS，并依据风险补充鉴权和限流。
"""

from fastapi import FastAPI

from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

import os
import time
import re
from pathlib import Path

from api.routes_qa import router as qa_router
from api.routes_docs import router as docs_router
from api.routes_eval import router as eval_router
from api.routes_ops import router as ops_router

from api.deps import check_secrets
from base.tracing import current_trace_id, reset_trace_id, set_trace_id
from base.logger import logger
from base.config import config

def create_app() -> FastAPI:
    """
    函数功能: 创建并装配 FastAPI 应用, 包括 CORS 与 trace_id 中间件、静态目录挂载、首页、业务路由和系统状态接口, 并执行密钥配置自检.
    :return: 装配完成的 FastAPI 应用实例
    """
    
    app = FastAPI(title="智学答问答系统", description="智学答教育资料智能问答系统")

    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        
        allow_credentials=True,     
        allow_methods=["*"],        
        allow_headers=["*"],        
    )

    @app.middleware("http")
    async def trace_requests(request, call_next):
        """为每次 HTTP 请求设置可跨异步调用读取的编号，并写入响应头和日志。"""
        supplied_id = request.headers.get("X-Trace-Id", "")
        safe_trace_id = re.sub(r"[^A-Za-z0-9_.-]", "", supplied_id)[:128]
        token = set_trace_id(safe_trace_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Trace-Id"] = current_trace_id()
            logger.info(
                "HTTP 请求完成 trace_id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
                current_trace_id(), request.method, request.url.path,
                response.status_code, (time.perf_counter() - started) * 1000,
            )
            return response
        finally:
            reset_trace_id(token)

    
    static_dir = Path(config.PROJECT_ROOT) / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    check_secrets()

    app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')

    
    @app.get("/")               
    async def read_root():
        """
        函数功能: 返回静态目录下的 index.html 作为首页.
        :return: 指向 static/index.html 的 FileResponse
        """
        return FileResponse(static_dir / "index.html")

    
    app.include_router(qa_router)
    app.include_router(docs_router)
    app.include_router(eval_router)
    app.include_router(ops_router)

    @app.get("/api/system/status")
    async def system_status():
        """返回启动和配置状态，不连接外部存储，也不显示密钥值。"""
        return {
            "application": "智学答问答系统",
            "status": "running",
            "configured": {
                "mysql_password": bool(config.MYSQL_PASSWORD),
                "dashscope_api_key": bool(config.DASHSCOPE_API_KEY),
                "redis_password": bool(config.REDIS_PASSWORD),
            },
            "note": "数据库、向量库和模型会在首次使用问答或管理功能时按需连接。",
        }

    return app

app = create_app()

if __name__ == "__main__":
    
    
    

    import uvicorn      
    import os

    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))

    
    
    uvicorn.run("api.server:app", host=host, port=port, reload=False)
