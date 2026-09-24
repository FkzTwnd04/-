"""
提供查询改写子包的导入入口，向 pipeline 暴露查询转换能力。
策略选择由 router 完成，具体把问题改写成一个或多个搜索文本由 rewriter.py 执
行。
"""

from .rewriter import QueryRewriter

__all__ = ["QueryRewriter"]
