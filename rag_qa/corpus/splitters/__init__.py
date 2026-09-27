"""
导出中文递归切分器和模型切分器，供 processor 通过统一入口选择。
模型切分器还需要本地模型及 modelscope 依赖，未安装时只有它不可用。
"""

from rag_qa.corpus.splitters.recursive import ChineseTextSplitter

try:
    from rag_qa.corpus.splitters.model import SemanticSplitter
except ImportError:
    SemanticSplitter = None

__all__ = ["ChineseTextSplitter", "SemanticSplitter"]
