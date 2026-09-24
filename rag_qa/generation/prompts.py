"""
集中定义 RAG 系统提示和用户问题模板，负责将问题、历史和检索上下文按约定格式交给模型。
提示词可以要求模型基于资料作答，但仅靠提示文字不能保证引用真实或拒答正确；
这些要求还需要结构化引用校验及评测支持。
"""

from langchain_core.prompts import PromptTemplate

class RAGPrompts:
    
    @staticmethod
    def rag_prompt():
        """
        函数功能: 创建要求基于检索证据回答并标注引用编号的 RAG 提示模板.
        :return: 输入 context、history、question、phone 后可格式化的 PromptTemplate
        """
        return PromptTemplate(
            template="""  
                你是一个严谨的问答助手。先判断问题是否能由给定资料支持，再组织简洁、通俗的回答。
                当上下文包含 [S1]、[S2] 等编号时，只能使用上下文支持的事实，并在对应陈述后标注编号，例如 [S1]。
                不得编造引用编号、来源、页码或上下文没有提供的事实。多个资料共同支持时可以标注多个编号。
                如果问题需要项目资料才能回答但上下文为空或证据不足，请明确回复“资料中没有足够信息回答这个问题”，不要猜测。
                对明确属于通用常识的问题，可以使用常识回答；不得把常识伪装成检索资料结论。

                对话历史: {history}
                上下文: {context}  
                问题: {question}  

                如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
                回答:  
                """,
            input_variables=["context", "history", "question", "phone"],
        )

    @staticmethod
    def general_prompt():
        """
        函数功能: 创建不依赖知识库的通用问答提示模板.
        :return: 输入 history 和 question 后可格式化的 PromptTemplate
        """
        return PromptTemplate(
            template="""
            你是一个通用中文助手。请直接、清楚地回答用户问题；如果不知道就说明不确定，
            不要假装检索过知识库，也不要编造资料来源。

            最近对话：
            {history}

            用户问题：{question}
            助手回答：
            """,
            input_variables=["history", "question"],
        )

    
    @staticmethod
    def hyde_prompt():
        """
        函数功能: 创建 HyDE 假设答案生成模板, 用生成文本增强后续检索.
        :return: 输入 query 后可格式化的 PromptTemplate
        """
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: {query}  
            假设答案:  
            """,
            input_variables=["query"],
        )

    
    @staticmethod
    def subquery_prompt():
        """
        函数功能: 创建复杂问题拆分模板, 要求模型逐行输出多个简单子查询.
        :return: 输入 query 后可格式化的 PromptTemplate
        """
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            查询: {query}  
            子查询:  
            """,
            input_variables=["query"],
        )

    
    @staticmethod
    def backtracking_prompt():
        """
        函数功能: 创建回溯改写模板, 将复杂问题改写为更基础、更易检索的问题.
        :return: 输入 query 后可格式化的 PromptTemplate
        """
        return PromptTemplate(
            template="""  
            将以下复杂查询简化为一个更简单的问题：  
            查询: {query}  
            简化问题:  
            """,
            input_variables=["query"],
        )

if __name__ == '__main__':
    
    rag_prompt = RAGPrompts.rag_prompt()
    result = rag_prompt.format(
        context="黑马程序员是一家IT培训机构, 主打Python, AI等课程",
        history="",
        question="这家机构的名字叫什么?",
        phone="13112345678",
    )
    print(result)
    print('-' * 30)

    
    hyde_prompt = RAGPrompts.hyde_prompt()
    print(hyde_prompt.format(query="如何培养孩子的专注力"))
