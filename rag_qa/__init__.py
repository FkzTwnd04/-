"""
作为 RAG 包的轻量导出入口，提供常用的 Chunk、检索、提示词和问答组件，减少调用方的导入
路径负担。
重资源系统 IntegratedQASystem 应由调用方显式导入并创建，避免仅导入 rag_
qa 就触发数据库、Redis 或模型初始化。
"""

from .schema.chunk import Chunk                    
from .retriever.hybrid import HybridRetriever      
from .generation.prompts import RAGPrompts         
from .pipeline.qa import RAGSystem                 

__all__ = ["Chunk", "HybridRetriever", "RAGPrompts", "RAGSystem"]
