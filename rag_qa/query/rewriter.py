"""
根据策略将原始问句转换成适合检索的查询集合，例如直接查询、HyDE、子查询和回溯改写。
输出会交给 retriever 搜索，多个查询的结果由上层流程汇总去重。
模型改写失败时应保留使用原问题的安全退路，并评估额外查询带来的成本。
"""

from typing import List

from base.logger import logger
from rag_qa.generation.prompts import RAGPrompts

class QueryRewriter:
    """查询改写器: 依赖注入大模型, 自身不持有任何模型客户端."""

    def __init__(self, llm):
        """
        :param llm: 大模型调用对象(LLMClient 或 callable), 由外部注入
        """
        from rag_qa.generation.llm import as_llm

        self.llm = as_llm(llm)

    
    def rewrite(self, query: str, mode: str = "direct") -> List[str]:
        """
        函数功能: 按指定模式改写查询.
        :param query: 原始查询
        :param mode: 改写模式 -> direct(不改写) / hyde / subquery / backtrack
        :return: 改写后的查询列表(字符串列表)
                 说明: 返回列表而不是字符串, 是因为 subquery 会产生多个查询; 其余模式返回长度为1的列表.
        """
        if mode == "hyde":
            return [self._hyde(query)]
        if mode == "subquery":
            return self._subqueries(query)
        if mode == "backtrack":
            return [self._backtrack(query)]
        return [query]

    
    def _hyde(self, query: str) -> str:
        prompt = RAGPrompts.hyde_prompt().format(query=query)
        hypo_answer = self.llm.complete(prompt).strip()
        
        if not hypo_answer:
            logger.warning("HyDE 生成假设答案失败, 降级为原查询")
            return query
        logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
        return hypo_answer

    
    def _subqueries(self, query: str) -> List[str]:
        prompt = RAGPrompts.subquery_prompt().format(query=query)
        text = self.llm.complete(prompt).strip()
        subqueries = [q.strip() for q in text.split("\n") if q.strip()]
        if not subqueries:
            logger.warning("未能生成有效的子查询, 降级为原查询")
            return [query]
        logger.info(f"生成的子查询: {subqueries}")
        return subqueries

    
    def _backtrack(self, query: str) -> str:
        prompt = RAGPrompts.backtracking_prompt().format(query=query)
        simplified = self.llm.complete(prompt).strip()
        if not simplified:
            logger.warning("回溯问题生成失败, 降级为原查询")
            return query
        logger.info(f"生成的回溯问题: '{simplified}'")
        return simplified
