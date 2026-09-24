"""
评测模块包入口。

对外提供 JSONL 样本读取、基础指标计算和 RAG 批量评测器；导入本包不会连接数据库、
Milvus 或模型服务，外部资源只在显式运行评测时按需初始化。
"""

from rag_qa.evaluation.dataset import EvaluationSample, load_dataset
from rag_qa.evaluation.metrics import aggregate_intent_metrics, aggregate_metrics
from rag_qa.evaluation.runner import RAGEvaluationRunner

__all__ = [
    "EvaluationSample",
    "load_dataset",
    "aggregate_metrics",
    "aggregate_intent_metrics",
    "RAGEvaluationRunner",
]
