"""
定义问答 API 的请求与响应字段，包括问题、来源筛选、会话编号、答案、运营反馈编号和引用。
这里是客户端协议的共享模型；路由新增响应字段时应同步填写并保持前后端含义一致。
"""

from pydantic import BaseModel, Field

from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    query: str                           
    source_filter: Optional[str] = None  
    session_id: Optional[str] = None     

class QueryResponse(BaseModel):
    answer: str                 
    is_streaming: bool          
    session_id: str             
    processing_time: float      
    request_id: Optional[int] = None  # 运营请求记录编号；客户端提交反馈时用它关联本次答案。
    hit_layer: Optional[str] = None  # 本次回答的来源层级，例如 intent、L1、L2a 或 L3。
    intent: Optional[str] = None  # 代表问句检索确认的意图标签；未命中时为空。
    citations: List[Dict[str, Any]] = Field(default_factory=list)  # RAG 使用的引用来源；没有文档引用时为空列表。
