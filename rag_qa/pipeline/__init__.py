"""
提供 pipeline 子包的公开入口，让 API 和命令行代码导入问答编排组件。
流程实现分别位于 qa.py 和 integrated.py；
包初始化本身不应启动问答或创建外部连接。
"""

from .qa import RAGSystem

__all__ = ["RAGSystem"]
