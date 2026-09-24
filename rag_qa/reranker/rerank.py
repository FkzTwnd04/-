"""
对初步检索出的查询—文本块候选调用重排模型，按相关性重新排序并返回上层需要的 Top-K。
重排通常比向量初筛更精细但计算代价更高，因此只处理有限候选；
应保留 Chunk 的来源、页码和标识信息，并通过评测决定开关与保留数量。
"""

import os
from typing import List, Optional

from base.config import config
from base.logger import logger
from rag_qa.schema.chunk import Chunk

class Reranker:
    """重排器: 输入 (查询, Chunk列表), 输出按相关性降序的 Chunk 列表(带 score)."""

    def __init__(self, model_path: str = None, device: str = None, enabled: bool = True):
        """
        :param model_path: 模型路径, 默认 rag_qa/models/bge-reranker-large
        :param device: 运行设备, 默认自动判断
        :param enabled: 是否启用 -> 关闭时 rerank 直接原样返回, 用于做"开/关重排"的对比实验
        """
        self.model_path = model_path or os.path.join(config.MODELS_DIR, "bge-reranker-large")
        self.device = device or self._detect_device()
        self.enabled = enabled
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
        函数功能: 延迟加载重排模型, 首次访问时用 model_path 和 device 创建 CrossEncoder 并缓存.
        :return: CrossEncoder 模型实例
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"正在加载重排模型: {self.model_path} (设备: {self.device})")
            self._model = CrossEncoder(self.model_path, device=self.device)
        return self._model

    
    def rerank(self, query: str, chunks: List[Chunk], top_k: Optional[int] = None) -> List[Chunk]:
        """
        函数功能: 按"查询-文档"相关性重新排序并截取.
        :param query: 查询文本
        :param chunks: 待重排的 Chunk 列表
        :param top_k: 返回前 top_k 个, None 表示不截取
        :return: 按得分降序的 Chunk 列表(每个 Chunk 的 score 被写成重排分)
        """
        
        if not chunks:
            return []
        if not self.enabled or len(chunks) < 2:
            return chunks[:top_k] if top_k else chunks

        
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)

        
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        
        ranked = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)]
        logger.info(f"重排完成, 候选 {len(chunks)} 个, 最高分 {ranked[0].score:.4f}")

        return ranked[:top_k] if top_k else ranked
