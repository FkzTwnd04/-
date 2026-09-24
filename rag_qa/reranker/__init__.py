"""
提供重排子包的统一导入入口，便于问答编排调用重排实现。
排序逻辑、模型加载和结果分数都由 rerank.py 管理。
"""

from rag_qa.reranker.rerank import Reranker

__all__ = ["Reranker"]
