"""
定义各格式加载器共用的基类：一个文件产出一个 LangChain Document，正文由子类的 extract_text 提供。
来源路径写入 metadata["source"]，processor 随后会补充学科来源、文件路径和时间戳。
"""

from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class FileTextLoader(BaseLoader):

    def __init__(self, file_path: str) -> None:
        """
        函数功能: 记录待解析的文件路径.
        :param file_path: 文件路径
        :return: None
        """
        self.file_path = str(file_path)

    def extract_text(self) -> str:
        """
        函数功能: 提取文件全文, 由具体格式的子类实现.
        :return: 文件正文字符串
        """
        raise NotImplementedError

    def lazy_load(self) -> Iterator[Document]:
        """
        函数功能: 解析文件并产出包含全文的单个 Document.
        :return: 生成器, 产出一个 Document
        """
        yield Document(page_content=self.extract_text(), metadata={"source": self.file_path})


def join_lines(lines) -> str:
    """
    函数功能: 去掉空白行后按换行拼接文本行.
    :param lines: 可迭代的文本行
    :return: 拼接后的字符串
    """
    return "\n".join(line.strip() for line in lines if line and line.strip())
