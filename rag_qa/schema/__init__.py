"""
对外导出问答链路共同使用的数据结构，当前以 Chunk 为主。
schema 子包保持轻量，避免导入时连接数据库或加载模型。
"""

from .chunk import Chunk

__all__ = ["Chunk"]
