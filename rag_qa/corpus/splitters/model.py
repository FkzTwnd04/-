"""
使用本地 ModelScope 中文文档分段模型判断语义边界，把长文切成主题相对完整的片段。
相比按字符长度切分，模型切分可能保留更完整的主题边界，但会增加模型加载和处理成本，
且需通过评测确认切分质量。模型文件需预先放到 rag_qa/models 下。
"""

import os
import re
from functools import lru_cache
from typing import List

from langchain_text_splitters import TextSplitter

from base.config import config

DEFAULT_MODEL_DIR = os.path.join(config.MODELS_DIR, "nlp_bert_document-segmentation_chinese-base")


@lru_cache(maxsize=2)
def _segmentation_pipeline(model_dir: str):
    """
    函数功能: 加载文档分段模型并缓存, 同一模型目录只加载一次.
    :param model_dir: 本地模型目录
    :return: ModelScope 分段 pipeline
    """
    from modelscope.pipelines import pipeline
    return pipeline(task="document-segmentation", model=model_dir, device="cpu")


class SemanticSplitter(TextSplitter):

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, flatten_whitespace: bool = False, **kwargs) -> None:
        """
        函数功能: 创建模型切分器.
        :param model_dir: 本地文档分段模型目录
        :param flatten_whitespace: 为 True 时先把所有空白压成单个空格(适合换行错乱的 PDF 文本)
        :param kwargs: 传给 TextSplitter 的其他参数
        :return: None
        """
        super().__init__(**kwargs)
        self.model_dir = model_dir
        self.flatten_whitespace = flatten_whitespace

    def split_text(self, text: str) -> List[str]:
        """
        函数功能: 用分段模型切分文本; 模型输出以 "\\n\\t" 分隔各段.
        :param text: 待切分的文本
        :return: 片段列表
        """
        if self.flatten_whitespace:
            text = re.sub(r"\s+", " ", text)
        result = _segmentation_pipeline(self.model_dir)(documents=text)
        return [part.strip() for part in result["text"].split("\n\t") if part.strip()]
