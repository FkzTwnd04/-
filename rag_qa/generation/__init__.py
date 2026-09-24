"""
为生成子包提供统一导出入口，聚合大模型客户端和提示词相关能力。
这里仅管理 Python 包的公开接口，具体请求格式、流式调用和提示模板分别由 llm.py 与
 prompts.py 负责。
"""

from .llm import LLMClient, CallableLLM, as_llm
from .prompts import RAGPrompts

__all__ = ["LLMClient", "CallableLLM", "as_llm", "RAGPrompts"]
