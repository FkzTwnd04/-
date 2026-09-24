"""
在文档解析和向量写入之前识别重复文件及内容更新。

指纹根据原始文件字节计算，避免改了切分参数后被误判为新文档；判断范围带租户，
同一文件只有在同一个租户下才可以跳过重复入库。
"""

from typing import Any, Optional, Tuple

from storage.documents import DocumentStore, content_hash


class DocumentDeduplicator:
    """把内容哈希判定委托给文档台账，避免重复维护第二份去重状态。"""

    def __init__(
        self,
        document_store: DocumentStore,  # 负责持久化内容指纹和来源路径的文档台账。
    ):
        self.document_store = document_store

    @staticmethod
    def content_hash(content: Any) -> str:
        """计算文件内容指纹；str 适用于文本，bytes 适用于上传文件原始内容。"""

        return content_hash(content)

    def classify(
        self,
        content: bytes,  # 上传文件的原始字节；直接对它计算 SHA-256，覆盖整个文件内容。
        source_path: str,  # 来源标识；同一路径内容改变时分类为 updated。
    ) -> Tuple[str, Optional[str]]:
        """返回 new、duplicate 或 updated，以及可能复用的 doc_id。"""

        return self.document_store.classify_content(source_path, content)
