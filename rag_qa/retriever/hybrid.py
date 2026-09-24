"""
协调稠密向量检索与稀疏检索，组合候选结果并处理来源筛选、父子块提升和去重，再交给上层排序或生成。
它是 Milvus 能力与问答编排之间的适配层；
租户条件必须由可信身份构造并在检索时下推，不能依赖返回后的临时过滤。
"""

from typing import List, Optional

from base.config import config
from base.logger import logger
from rag_qa.retriever.base import BaseRetriever
from rag_qa.retriever.embedding import BGEM3Embedder
from rag_qa.retriever.milvus_store import MilvusStore
from rag_qa.reranker.rerank import Reranker
from rag_qa.schema.chunk import Chunk

class HybridRetriever(BaseRetriever):
    """混合检索器: 稠密 + 稀疏双路召回, 父块提升去重, 重排后返回."""

    def __init__(
        self,
        store: MilvusStore = None,
        embedder: BGEM3Embedder = None,
        reranker: Reranker = None,
        candidate_m: int = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.7,
    ):
        """
        :param store: 向量库, 为空时按配置自动创建 MilvusStore
        :param embedder: 向量化器, 为空时自动创建 BGEM3Embedder
        :param reranker: 重排器, 为空时自动创建 Reranker
        :param candidate_m: 最终返回的文档块数(送进上下文的数量)
        :param dense_weight/sparse_weight: 双路召回的融合权重
        """
        
        self.embedder = embedder or BGEM3Embedder()
        self.store = store or MilvusStore(dense_dim=None)
        self.reranker = reranker or Reranker()

        
        self.candidate_m = candidate_m or config.CANDIDATE_M
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    
    def search(self, query: str, k: Optional[int] = None, filters: Optional[dict] = None) -> List[Chunk]:
        """
        函数功能: 执行一次完整的混合检索.
        :param query: 查询文本
        :param k: 粗召回数量
        :param filters: 过滤条件字典, 例如 {'source': 'ai', 'tenant_id': 't1'}
        :return: 重排后的 Chunk 列表(最多 candidate_m 个)
        """
        k = k or config.RETRIEVAL_K

        
        dense_vector, sparse_vector = self.embedder.embed_one(query)

        
        hits = self.store.hybrid_search_raw(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            k=k,
            filters=filters,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )
        logger.info(f"混合检索命中 {len(hits)} 个子块 (k={k}, filters={filters})")

        
        # 保留 Milvus hit 外层的主键 ID；entity 通常只包含显式请求的业务字段。
        sub_chunks = [Chunk.from_milvus_hit(hit) for hit in hits]

        
        parent_chunks = self._to_unique_parents(sub_chunks)

        
        return self.reranker.rerank(query, parent_chunks, top_k=self.candidate_m)

    
    def hybrid_search_with_rerank(self, query: str, k: Optional[int] = None, source_filter: str = None) -> List[Chunk]:
        """
        函数功能: 兼容旧调用签名, 内部转成新的 filters 字典.
        :param source_filter: 旧语义是学科(ai/java), 新语义是文档来源; 传 None 表示不过滤
        :return: Chunk 列表(注意: 旧版本返回 langchain Document, 现统一为 Chunk)
        """
        filters = {"source": source_filter} if source_filter else None
        return self.search(query, k=k, filters=filters)

    
    @staticmethod
    def _to_unique_parents(sub_chunks: List[Chunk]) -> List[Chunk]:
        """
        函数功能: 子块命中后返回其父块, 并按父块去重.
        为什么要返回父块: 子块切得小(300字)是为了精准匹配, 但送给大模型时需要完整上下文,
                        直接把 300 字塞进去常常语义不完整, 所以"用子块检索, 用父块生成".
        为什么会重复: 同一父块下的多个子块可能同时被召回.
        """
        seen = set()
        parents: List[Chunk] = []

        for chunk in sub_chunks:
            
            if not chunk.parent_text:
                key = chunk.dedup_key
                if key in seen:
                    continue
                seen.add(key)
                parents.append(chunk)
                continue

            
            key = chunk.parent_id or chunk.parent_text
            if key in seen:
                continue
            seen.add(key)

            parents.append(
                Chunk(
                    text=chunk.parent_text,
                    chunk_id=chunk.parent_id or key,
                    doc_id=chunk.doc_id,
                    source=chunk.source,
                    page=chunk.page,
                    publish_time=chunk.publish_time,
                    tenant_id=chunk.tenant_id,
                    parent_id=chunk.parent_id,
                    parent_text=chunk.parent_text,
                    score=chunk.score,
                    filename=chunk.filename,
                )
            )

        logger.info(f"父块提升去重: {len(sub_chunks)} 个子块 -> {len(parents)} 个父块")
        return parents

    
    def add(self, chunks) -> int:
        """
        函数功能: 把文档块交给 MilvusStore 写入, 并使用本检索器的 embedder 计算向量.
        :param chunks: Chunk 列表 或 langchain Document 列表
        :return: 写入条数
        """
        return self.store.add(chunks, embedder=self.embedder)

    def delete_document(self, doc_id: str, tenant_id: str = "default") -> int:
        """把按文档删除操作转交给 Milvus 存储，并始终带租户过滤条件。"""

        return self.store.delete_document(doc_id, tenant_id=tenant_id)
