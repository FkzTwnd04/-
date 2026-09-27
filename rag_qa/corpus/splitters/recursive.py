"""
按“段落 → 换行 → 句末标点 → 分号 → 逗号 → 单字”的优先级递归拆分中文长文本，并控制块大小和重叠区域。
切分结果决定向量索引粒度；块太大可能混入多个主题，块太小可能缺少回答所需上下文。
调整参数后应重新建立索引并比较召回效果。
"""

import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 从粗到细的切分边界（正则）；标点保留在前一段末尾，句子不会以标点开头。
CHINESE_SEPARATORS = [
    r"\n{2,}",
    r"\n",
    r"[。！？]|[.!?](?=\s)",
    r"[；;]",
    r"[，,、]",
    "",
]


class ChineseTextSplitter(RecursiveCharacterTextSplitter):

    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs) -> None:
        """
        函数功能: 创建使用中文标点边界的递归切分器, 其余参数原样传给 RecursiveCharacterTextSplitter.
        :param chunk_size: 单块最大字符数
        :param chunk_overlap: 相邻块重叠字符数
        :return: None
        """
        kwargs.setdefault("separators", CHINESE_SEPARATORS)
        kwargs.setdefault("is_separator_regex", True)
        kwargs.setdefault("keep_separator", "end")
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    def split_text(self, text: str) -> List[str]:
        """
        函数功能: 切分文本, 并把块内连续空行压成单个换行、去掉首尾空白和空块.
        :param text: 待切分的文本
        :return: 文本块列表
        """
        chunks = (re.sub(r"\n{2,}", "\n", chunk).strip() for chunk in super().split_text(text))
        return [chunk for chunk in chunks if chunk]
