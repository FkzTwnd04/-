"""
描述检索器需要提供的基本操作，使问答编排面向统一接口而不是直接依赖某个向量数据库的实现细节。
新增或替换后端时应保持输入、过滤参数和返回 Chunk 的约定一致。
"""

from typing import List, Optional

from rag_qa.schema.chunk import Chunk

class BaseRetriever:
    """检索器接口: 输入查询, 输出 Chunk 列表."""

    
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        函数功能: 执行检索.
        :param query: 查询文本(可以是原始问题, 也可以是改写后的查询)
        :param k: 召回数量, None 表示用配置默认值
        :param filters: 过滤条件字典, 例如: {'source': 'ai', 'tenant_id': 't1'}
                        注意: 过滤必须在这里"下推"到检索引擎, 而不是召回后再用 Python 过滤.
        :return: Chunk 列表
        """
        raise NotImplementedError

    
    def search_batch(
        self,
        queries,
        k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        函数功能: 对多个查询分别检索并合并(子查询策略会用到).
        :param queries: 查询列表
        :return: 合并后的 Chunk 列表(未去重, 去重是编排层的职责)
        """
        results: List[Chunk] = []
        for query in queries:
            results.extend(self.search(query, k=k, filters=filters) or [])
        return results

    
    def add(self, chunks) -> int:
        """
        函数功能: 把文档块写入索引.
        :param chunks: Chunk 列表 或 langchain Document 列表
        :return: 写入的条数
        """
        raise NotImplementedError
