"""
导出中文递归切分器和模型辅助切分器，供 processor 通过统一入口选择。
实际使用哪种切分器由处理代码和配置决定；
引入模型切分器还需要准备本地模型及其运行依赖。
"""

from .recursive import *

try:
    from .model import *
except ImportError:
    pass