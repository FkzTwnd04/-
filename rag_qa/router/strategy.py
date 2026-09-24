"""
根据进入 RAG 的问题及分类结果选择后续检索策略，并对策略名称做归一化，供 query.rew
riter 执行。
它决定“如何搜索知识库”，而 intent.py 负责“是否属于某类业务意图”；
意图检索命中不等于检索策略已经选定。
"""

from base.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

STRATEGY_DIRECT = "直接检索"
STRATEGY_HYDE = "假设问题检索"
STRATEGY_SUBQUERY = "子查询检索"
STRATEGY_BACKTRACK = "回溯问题检索"

STRATEGY_TO_REWRITE_MODE = {
    STRATEGY_DIRECT: "direct",
    STRATEGY_HYDE: "hyde",
    STRATEGY_SUBQUERY: "subquery",
    STRATEGY_BACKTRACK: "backtrack",
}

class StrategySelector:
    """检索策略选择器: 只做决策, 不持有模型客户端(由外部注入)."""

    def __init__(self, llm=None):
        """
        函数功能: 统一模型接口并装配用于选择检索策略的 LCEL 链.
        :param llm: 大模型调用对象, 为空时自动创建一个默认客户端
        :return: None
        """
        from rag_qa.generation.llm import as_llm, as_runnable

        self.llm = as_llm(llm)
        self.strategy_prompt_template = self._get_strategy_prompt()
        self.strategy_chain = (
            self.strategy_prompt_template
            | as_runnable(self.llm)
            | StrOutputParser()
        )

    
    def select_strategy(self, query: str) -> str:
        """
        函数功能: 为查询选择检索策略.
        :param query: 查询文本
        :return: 标准策略名(四个常量之一)
        """
        try:
            raw = self.strategy_chain.invoke({"query": query})
            strategy = self._normalize(raw)
        except Exception as e:
            logger.error(f"策略选择失败, 降级为直接检索: {e}")
            strategy = STRATEGY_DIRECT
        logger.info(f"为查询 '{query}' 选择的检索策略: {strategy}")
        return strategy

    
    @staticmethod
    def _normalize(raw: str) -> str:
        """
        函数功能: 大模型经常返回带解释/括号的文本, 用关键字匹配收敛成标准策略名.
        :param raw: 大模型原始输出
        :return: 标准策略名
        """
        text = (raw or "").strip()
        if "假设" in text or "hyde" in text.lower():
            return STRATEGY_HYDE
        if "子查询" in text:
            return STRATEGY_SUBQUERY
        if "回溯" in text:
            return STRATEGY_BACKTRACK
        return STRATEGY_DIRECT

    
    def call_dashscope(self, prompt: str) -> str:
        """
        函数功能: 兼容旧调用入口, 把提示词直接交给统一模型客户端.
        :param prompt: 已格式化的提示词文本
        :return: 大模型一次性生成的文本
        """
        return self.llm.complete(prompt)

    
    def _get_strategy_prompt(self):
        """
        函数功能: 创建让模型从直接、HyDE、子查询和回溯中选择一种策略的提示模板.
        :return: 输入 query 后可格式化的 PromptTemplate
        """
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题。
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题。
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题。
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题。

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """,
            input_variables=["query"],
        )

if __name__ == '__main__':
    ss = StrategySelector()
    print(ss.select_strategy('对比北京和上海的人才引进政策, 从补贴金额, 落户难度, 产业适配性三个方面分析'))
