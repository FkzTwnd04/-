"""
加载本地嵌入模型，把查询或文档块转换成用于相似度搜索的向量，并提供稀疏检索所需表示。
模型目录、向量维度和 Milvus 集合 schema 必须匹配；
更换模型时旧向量不能默认与新向量混用。
"""

import os
from typing import List, Sequence, Tuple

from base.config import config
from base.logger import logger

class BGEM3Embedder:
    """文本向量化: 对外只暴露 embed() 和 embed_one(), 内部模型延迟加载."""

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        use_fp16: bool = None,
    ):
        """
        :param model_path: 模型本地路径, 默认取配置 rag_qa/models/bge-m3
        :param device: 运行设备(cuda/cpu), 默认自动判断
        :param use_fp16: 是否半精度, 默认 GPU 时开启
        """
        
        self.model_path = model_path or os.path.join(config.MODELS_DIR, "bge-m3")

        
        self.device = device or self._detect_device()

        
        self.use_fp16 = use_fp16 if use_fp16 is not None else (self.device == "cuda")

        
        self._model = None

    
    @staticmethod
    def _detect_device() -> str:
        """
        函数功能: 检测 CUDA 是否可用以选择运行设备; 导入 torch 或检测出错时降级为 cpu.
        :return: "cuda" 或 "cpu"
        """
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            
            return "cpu"

    
    @property
    def model(self):
        """
        函数功能: 延迟加载 BGE-M3 模型, 首次访问时按 model_path、use_fp16 和 device 创建 BGEM3EmbeddingFunction 并缓存.
        :return: BGEM3EmbeddingFunction 实例
        """
        if self._model is None:
            from milvus_model.hybrid import BGEM3EmbeddingFunction

            logger.info(f"正在加载 BGE-M3 模型: {self.model_path} (设备: {self.device})")
            self._model = BGEM3EmbeddingFunction(
                model_name_or_path=self.model_path,
                use_fp16=self.use_fp16,
                device=self.device,
            )
        return self._model

    
    @property
    def dense_dim(self) -> int:
        """
        函数功能: 获取模型稠密向量的维度(访问时会触发模型加载).
        :return: 稠密向量维度
        """
        return self.model.dim["dense"]

    
    def embed(self, texts: Sequence[str]) -> dict:
        """
        函数功能: 把一批文本转成向量.
        :param texts: 文本列表
        :return: 字典, 形如 {'dense': [[...], ...], 'sparse': 稀疏矩阵}
        """
        return self.model(list(texts))

    
    def embed_one(self, text: str) -> Tuple[List[float], dict]:
        """
        函数功能: 把单条查询文本转成 Milvus 可直接用的格式.
        :param text: 查询文本
        :return: (dense 列表, sparse 字典{词表索引: 权重})
        """
        embeddings = self.embed([text])
        dense_vector = list(embeddings["dense"][0])
        sparse_vector = self._sparse_row_to_dict(embeddings["sparse"], 0)
        return dense_vector, sparse_vector

    
    @staticmethod
    def _sparse_row_to_dict(sparse_matrix, row_index: int) -> dict:
        """
        函数功能: 取稀疏矩阵的第 row_index 行, 转成字典.
        说明: 稀疏矩阵只存非零值(indices=位置, data=权重), 零值不存, 所以几千维的向量实际只有几十个有效项.
        """
        row = sparse_matrix[[row_index]]
        return {int(idx): float(val) for idx, val in zip(row.indices, row.data)}
