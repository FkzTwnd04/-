"""
提供问题路由子包的统一导入入口。
子包中分别有简单意图规则、BERT 类别分类和检索策略选择；
这些能力的输入、输出及成本不同，不应在文档中混称为一种分类功能。
"""

from .classifier import QueryClassifier
from .intent_retriever import IntentRetriever
from .strategy import StrategySelector

__all__ = ["QueryClassifier", "IntentRetriever", "StrategySelector"]
