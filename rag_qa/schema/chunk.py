"""
定义 Chunk 文档块结构，保存正文、文档及块标识、来源、页码、得分、租户字段和父子块关系，并
提供与框架文档对象的转换。
retriever、reranker、generation 等模块可通过该结构传递资料。
字段加入后还需检查 Milvus schema 和 API 输出；
tenant_id 字段本身不构成已完成的租户隔离。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import hashlib
import os

@dataclass
class Chunk:
    """检索链路的标准数据单元: 检索, 重排, 生成, 评测 四层都只认这个结构."""

    
    text: str                       

    
    doc_id: str = ""                
    chunk_id: str = ""              
    source: str = ""                
    page: Optional[int] = None      

    
    score: float = 0.0              

    
    publish_time: str = ""          
    tenant_id: str = ""             

    
    parent_id: str = ""             
    parent_text: str = ""           

    
    extra: Dict[str, Any] = field(default_factory=dict)
    filename: str = ""               # 原始知识文件名；用于回答展示实际引用的文件，而不是来源类别标签。

    
    @property
    def dedup_key(self) -> str:
        if self.chunk_id:
            return self.chunk_id
        if self.doc_id:
            return self.doc_id
        return hashlib.md5(self.text.encode("utf-8")).hexdigest()

    
    def to_document(self):
        """
        函数功能: 把 Chunk 转成 langchain Document, 供未改造的旧模块使用.
        :return: Document 对象
        """
        from langchain_core.documents import Document

        return Document(
            page_content=self.text,
            metadata={
                "doc_id": self.doc_id,
                "parent_id": self.parent_id,
                "parent_content": self.parent_text,
                "source": self.source,
                "timestamp": self.publish_time,
                "tenant_id": self.tenant_id,
                "page": self.page,
                "chunk_id": self.chunk_id,
                "filename": self.filename,
            },
        )

    
    @classmethod
    def from_document(cls, doc) -> "Chunk":
        """
        函数功能: 兼容旧链路, 把 Document 提升为 Chunk.
        :param doc: langchain Document 对象
        :return: Chunk 对象
        """
        meta = getattr(doc, "metadata", {}) or {}
        return cls(
            text=doc.page_content,
            doc_id=meta.get("doc_id", "") or "",
            chunk_id=meta.get("chunk_id", ""),
            source=meta.get("source", "") or "",
            publish_time=meta.get("timestamp", "") or "",
            tenant_id=meta.get("tenant_id", "") or "",
            page=meta.get("page"),
            parent_id=meta.get("parent_id", "") or "",
            parent_text=meta.get("parent_content", "") or "",
            score=meta.get("score", 0.0) or 0.0,
            filename=meta.get("filename") or os.path.basename(meta.get("file_path", "") or ""),
        )

    
    @classmethod
    def from_milvus_hit(cls, hit: dict) -> "Chunk":
        """
        函数功能: 把 Milvus 检索命中转成 Chunk，并保留稳定的主键 ID.
        :param hit: Milvus 返回的 entity 字典，或同时包含 id 和 entity 的命中对象
        :return: Chunk 对象
        """
        # Milvus 搜索结果常把主键放在外层 hit.id、业务字段放在 hit.entity；
        # 统一合并后，评测标注和检索结果就能使用同一个稳定 chunk_id。
        outer_id = hit.get("id", "")
        entity = hit.get("entity", hit)
        if not isinstance(entity, dict):
            entity = {}
        if not entity.get("id") and outer_id:
            entity = {**entity, "id": outer_id}

        return cls(
            text=entity.get("text", "") or "",
            doc_id=entity.get("doc_id", "") or "",
            chunk_id=entity.get("id", "") or "",
            source=entity.get("source", "") or "",
            publish_time=entity.get("timestamp", "") or "",
            tenant_id=entity.get("tenant_id", "") or "",
            page=entity.get("page"),
            parent_id=entity.get("parent_id", "") or "",
            parent_text=entity.get("parent_content", "") or "",
            filename=entity.get("filename", "") or "",
        )

    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "score": self.score,
            "publish_time": self.publish_time,
            "tenant_id": self.tenant_id,
            "parent_id": self.parent_id,
            "parent_text": self.parent_text,
            "filename": self.filename,
        }

def to_chunk(item) -> Chunk:
    """
    函数功能: 把任意形态的文档块统一成 Chunk.
    存在的意义: 重构过渡期里, 有的模块已经改造完(返回 Chunk), 有的还没改(返回 langchain Document),
               编排层不该关心对方改造到哪一步, 统一在这里收敛.
    :param item: Chunk 对象 或 langchain Document 对象 或 字典
    :return: Chunk 对象
    """
    if isinstance(item, Chunk):
        return item
    if hasattr(item, "page_content"):
        return Chunk.from_document(item)
    if isinstance(item, dict):
        return Chunk.from_milvus_hit(item)
    raise TypeError(f"无法转换为 Chunk 的类型: {type(item)}")

def docs_to_chunks(docs) -> list:
    """把 Document 列表批量转成 Chunk 列表."""
    return [to_chunk(doc) for doc in docs]

def chunks_to_docs(chunks) -> list:
    """把 Chunk 列表批量转成 Document 列表(兼容旧模块)."""
    return [chunk.to_document() for chunk in chunks]
