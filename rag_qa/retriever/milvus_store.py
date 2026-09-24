"""
封装 Milvus 集合的连接、schema 初始化、文档向量写入和底层搜索操作，并把结果转成问
答流程使用的数据结构。
运行依赖可用的 Milvus 服务与匹配的模型维度；
写入和读取字段需要与 Chunk 定义保持同步。
"""

import hashlib
from typing import List, Optional

from base.config import config
from base.logger import logger

def build_filter_expr(filters: Optional[dict]) -> str:
    """
    函数功能: 把过滤条件字典拼成 Milvus 的过滤表达式.
    :param filters: 例如 {'source': 'ai', 'tenant_id': 't1'}
    :return: 表达式字符串, 例如: source == 'ai' and tenant_id == 't1'; 无条件时返回空串
    """
    if not filters:
        return ""

    parts = []
    for key, value in filters.items():
        if value is None or value == "":
            continue
        
        safe_value = str(value).replace("\\", "\\\\").replace("'", "\\'")
        parts.append(f"{key} == '{safe_value}'")

    return " and ".join(parts)

class MilvusStore:
    """Milvus 集合管理 + 写入 + 原始检索(对外只返回 dict, 不返回 Chunk)."""

    def __init__(
        self,
        collection_name: str = None,
        host: str = None,
        port: int = None,
        database: str = None,
        client=None,
        dense_dim: int = None,
    ):
        """
        :param collection_name: 集合名, 默认取配置
        :param host/port/database: Milvus 连接信息, 默认取配置
        :param client: 可注入的 MilvusClient(测试时用来替换真实连接)
        :param dense_dim: 稠密向量维度, 建集合时需要; 已存在的集合不需要
        """
        
        self.collection_name = collection_name or config.MILVUS_COLLECTION_NAME
        self.host = host or config.MILVUS_HOST
        self.port = port or config.MILVUS_PORT
        self.database = database or config.MILVUS_DATABASE_NAME
        self.dense_dim = dense_dim

        
        if client is not None:
            self.client = client
        else:
            from pymilvus import MilvusClient

            self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)

        
        self._create_or_load_collection()

    
    def _create_or_load_collection(self):
        from pymilvus import DataType

        if not self.client.has_collection(self.collection_name):
            
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR,
                             dim=self.dense_dim or 1024)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)
            
            schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="page", datatype=DataType.INT64)

            
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector", index_name="dense_index",
                index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128},
            )
            index_params.add_index(
                field_name="sparse_vector", index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )

            self.client.create_collection(
                collection_name=self.collection_name, schema=schema, index_params=index_params
            )
            logger.info(f"已创建集合 {self.collection_name}")
        else:
            logger.info(f"已加载集合 {self.collection_name}")

        
        self.client.load_collection(self.collection_name)

    
    def add(self, chunks, embedder=None) -> int:
        """
        函数功能: 把文档块写入向量库(已向量化好的直接写, 没向量化的调用 embedder 现算).
        :param chunks: Chunk 列表 或 langchain Document 列表
        :param embedder: 向量化器(提供 embed(texts) 方法), 为空时假定 chunks 已带向量
        :return: 写入条数
        """
        from rag_qa.schema.chunk import to_chunk

        
        normalized = [to_chunk(item) for item in chunks]
        if not normalized:
            return 0

        
        texts = [chunk.text for chunk in normalized]
        embeddings = embedder.embed(texts) if embedder else None

        
        data = []
        for i, chunk in enumerate(normalized):
            row = {
                
                "id": chunk.chunk_id or hashlib.md5(chunk.text.encode("utf-8")).hexdigest(),
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "parent_id": chunk.parent_id,
                "parent_content": chunk.parent_text,
                "source": chunk.source or "unknown",
                "filename": chunk.filename or "",
                "timestamp": chunk.publish_time or "unknown",
                "tenant_id": chunk.tenant_id or "",
                "page": chunk.page if chunk.page is not None else 0,
            }
            if embeddings is not None:
                row["dense_vector"] = embeddings["dense"][i]
                row["sparse_vector"] = self._sparse_row_to_dict(embeddings["sparse"], i)
            data.append(row)

        
        self.client.upsert(collection_name=self.collection_name, data=data)
        logger.info(f"已插入/更新 {len(data)} 条数据到集合 {self.collection_name}")
        return len(data)

    
    def hybrid_search_raw(
        self,
        dense_vector,
        sparse_vector: dict,
        k: int = None,
        filters: dict = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.7,
    ) -> List[dict]:
        """
        函数功能: 执行稠密 + 稀疏混合检索.
        :param dense_vector: 查询稠密向量
        :param sparse_vector: 查询稀疏向量字典
        :param k: 召回数量
        :param filters: 过滤条件字典(在这里下推给 Milvus)
        :param dense_weight/sparse_weight: 两路结果的融合权重
        :return: 命中列表, 元素是含 'entity' 和 'distance' 的字典
        """
        from pymilvus import AnnSearchRequest, WeightedRanker

        k = k or config.RETRIEVAL_K
        filter_expr = build_filter_expr(filters)

        
        dense_request = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr,
        )

        
        sparse_request = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr,
        )

        
        ranker = WeightedRanker(dense_weight, sparse_weight)

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "doc_id", "parent_id", "parent_content", "source", "filename", "timestamp", "tenant_id", "page"],
        )
        
        return results[0] if results else []

    def delete_document(self, doc_id: str, tenant_id: str = "default") -> int:
        """删除指定租户、指定文档的全部向量块，防止更新文件后检索到旧内容。"""

        safe_doc_id = str(doc_id).replace("\\", "\\\\").replace('"', '\\"')
        safe_tenant_id = str(tenant_id).replace("\\", "\\\\").replace('"', '\\"')
        result = self.client.delete(
            collection_name=self.collection_name,
            filter=f'doc_id == "{safe_doc_id}" and tenant_id == "{safe_tenant_id}"',
        )
        return int(result.get("delete_count", 0)) if isinstance(result, dict) else 0

    
    @staticmethod
    def _sparse_row_to_dict(sparse_matrix, row_index: int) -> dict:
        row = sparse_matrix[[row_index]]
        return {int(idx): float(val) for idx, val in zip(row.indices, row.data)}
