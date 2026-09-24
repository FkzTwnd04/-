"""
定义会话创建、问答请求、历史读取/清除以及 WebSocket 流式回答等 HTTP 接口。
路由负责读取和校验请求、调用依赖中的问答系统并组织响应，不实现 BM25、向量检索、数据库 SQ
L 或提示词。
同步模型调用在异步路由中可能阻塞事件循环，部署并发服务时需用线程池或异步客户端处理。
"""

from fastapi import APIRouter, WebSocket, HTTPException, Query, Depends

from starlette.websockets import WebSocketDisconnect

import json

import uuid

import time

import asyncio

from api.schema import QueryRequest, QueryResponse

from api.deps import get_qa_system

from rag_qa.router.intent import dispatch
from base.logger import logger
from base.config import config
from base.tracing import reset_trace_id, set_trace_id

router = APIRouter()

@router.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())  
    return {"session_id": session_id}  

@router.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        
        history = get_qa_system().get_session_history(session_id)
        
        
        return {"session_id": session_id, "history": history}
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")

@router.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    
    success = get_qa_system().clear_session_history(session_id)
    if success:
        
        return {"status": "success", "message": "历史记录已清除"}
    else:
        
        raise HTTPException(status_code=500, detail="清除历史记录失败")

@router.post("/api/query")
async def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    start_time = time.time()  
    
    session_id = request.session_id or str(uuid.uuid4())
    
    intent, greeting_response = dispatch(request.query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "answer_mode": "greeting",
            "session_id": session_id,
            "processing_time": time.time() - start_time,
        }

    qa_system = get_qa_system()
    answer, hit_layer, intent_match = qa_system.lookup_reusable_answer(request.query)
    if not answer:
        
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    
    request_id = qa_system.ops_store.log_request(
        query=request.query, answer=answer, hit_layer=hit_layer or "L3",
        intent=intent_match.get("intent") if intent_match else None,
        latency_ms=int((time.time() - start_time) * 1000),
    )
    if session_id:
        qa_system.update_session_history(session_id, request.query, answer)
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time,
        "hit_layer": hit_layer,
        "answer_mode": "database" if hit_layer and hit_layer.startswith("jpkb_") else "greeting",
        "intent": intent_match.get("intent") if intent_match else None,
        "request_id": request_id,
    }

@router.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  
    trace_token = set_trace_id()
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info("WebSocket connected client=%s", client_host)
    try:
        while True:
            
            data = await websocket.receive_text()
            request_data = json.loads(data)  
            
            query = request_data.get("query")
            if not isinstance(query, str) or not query.strip():
                await websocket.send_json({"type": "error", "error": "query 必须是非空字符串"})
                continue
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            start_time = time.time()  
            
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })

            intent, greeting_response = dispatch(query)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({"type": "token", "token": greeting_response, "session_id": session_id})
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time,
                    })
                continue
            
            collected_answer = ""
            request_ids = []
            response_metadata = {}
            for token, is_complete in get_qa_system().query(
                query, source_filter=source_filter, session_id=session_id,
                request_id_callback=request_ids.append,
                metadata_callback=response_metadata.update,
            ):
                collected_answer += token  
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time,
                            "request_id": request_ids[0] if request_ids else None,
                            "answer_mode": response_metadata.get("answer_mode"),
                            "citations": response_metadata.get("citations", []),
                        })
                    break
                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })
                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time,
                            "request_id": request_ids[0] if request_ids else None,
                            "answer_mode": response_metadata.get("answer_mode"),
                            "citations": response_metadata.get("citations", []),
                        })
                    break
                await asyncio.sleep(0.01)  
    except WebSocketDisconnect as e:
        logger.info("WebSocket disconnected code=%s reason=%s", e.code, e.reason)
    except Exception as e:
        logger.exception("WebSocket request failed: %s", e)
        if websocket.client_state == websocket.client_state.CONNECTED:
            
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                
                await websocket.close()
        except Exception:
            logger.exception("Failed to close WebSocket cleanly")
        logger.info("WebSocket connection closed client=%s", client_host)
        reset_trace_id(trace_token)

@router.get("/health")
async def health_check():
    
    return {"status": "healthy"}  

@router.get("/api/sources")
async def get_sources():
    """从本地配置返回知识来源选项，不为展示页面初始化数据库或模型。"""
    return {"sources": config.VALID_SOURCES}
