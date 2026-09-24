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
    """
    函数功能: 生成一个新的 UUID 作为会话编号, 不写入数据库.
    :return: JSON 对象 {"session_id": 新会话编号字符串}
    """
    session_id = str(uuid.uuid4())  
    return {"session_id": session_id}  

@router.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """
    函数功能: 通过共享问答系统读取指定会话的历史记录, 任何异常都转换为 500 错误.
    :param session_id: 要查询的会话编号
    :return: JSON 对象 {"session_id": 会话编号, "history": 对话记录列表}
    """
    try:
        
        history = get_qa_system().get_session_history(session_id)
        
        
        return {"session_id": session_id, "history": history}
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")

@router.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    """
    函数功能: 清除指定会话的历史记录, 清除失败时返回 500 错误.
    :param session_id: 要清除历史的会话编号
    :return: JSON 对象 {"status": "success", "message": "历史记录已清除"}
    """
    
    success = get_qa_system().clear_session_history(session_id)
    if success:
        
        return {"status": "success", "message": "历史记录已清除"}
    else:
        
        raise HTTPException(status_code=500, detail="清除历史记录失败")

@router.post("/api/query")
async def query(request: QueryRequest):
    """
    函数功能: 处理非流式问答请求, 问候语或命中 jpkb 热点问答时直接返回答案并记录日志与会话历史, 否则提示改用 WebSocket 流式接口; query 为空时返回 400 错误.
    :param request: 问答请求体, 包含 query 和可选的 session_id
    :return: JSON 对象; 问候时含 answer/is_streaming=False/answer_mode="greeting"/session_id/processing_time, 未命中时含提示 answer/is_streaming=True/session_id/processing_time, 命中热点问答时另含 hit_layer/answer_mode/intent/request_id
    """
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
    """
    函数功能: 在 WebSocket 连接上循环接收 JSON 问题, 问候语直接回复, 其他问题调用问答系统逐 token 流式推送答案; 出现异常时发送 error 消息, 结束时关闭连接并恢复 trace_id.
    :param websocket: 客户端 WebSocket 连接, 每条消息为含 query/可选 source_filter/session_id 的 JSON 文本
    :return: None; 通过连接发送 start(含 session_id)、token(含 token/session_id)、end(含 session_id/is_complete/processing_time, 问答路径另含 request_id/answer_mode/citations)、error(含 error) 类型消息
    """
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
    """
    函数功能: 健康检查接口, 固定返回健康状态, 不检查数据库或模型等依赖.
    :return: JSON 对象 {"status": "healthy"}
    """
    
    return {"status": "healthy"}  

@router.get("/api/sources")
async def get_sources():
    """从本地配置返回知识来源选项，不为展示页面初始化数据库或模型。"""
    return {"sources": config.VALID_SOURCES}
