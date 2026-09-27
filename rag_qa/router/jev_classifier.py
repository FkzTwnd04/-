"""调用 TypeSafe Jev，将用户问题分到完整 RAG 或通用大模型回答路径。

Jev 只输出预先定义的路由选项及置信度，不负责检索资料或撰写答案。专业领域包括
RAG、Agent、人工智能及其工程实践；置信度不足或服务调用失败时，安全地选择完整 RAG。
"""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from base.config import config
from base.logger import logger


class JevIntentClassifier:
    """使用 Jev 的 Choice 输出在专业 RAG 与通用 LLM 之间进行路由。"""

    PROFESSIONAL = "professional_rag"
    GENERAL = "general_llm"

    def __init__(self, api_key=None, model=None, timeout=None, confidence_threshold=None, endpoint=None):
        """
        函数功能: 初始化 Jev 路由分类器的 API 密钥、模型、接口地址、超时和置信度阈值, 未传入的参数取配置默认值.
        :param api_key: Jev API 密钥, None 时取 config.JEV_API_KEY
        :param model: Jev 模型名, 未传时取 config.JEV_MODEL
        :param timeout: 请求超时秒数, 未传时取 config.JEV_TIMEOUT_SECONDS
        :param confidence_threshold: 路由置信度下限, None 时取 config.JEV_CONFIDENCE_THRESHOLD
        :param endpoint: System One 接口地址, 未传时取 config.JEV_ENDPOINT
        :return: None
        """
        self.api_key = api_key if api_key is not None else config.JEV_API_KEY
        self.model = model or config.JEV_MODEL
        self.endpoint = endpoint or config.JEV_ENDPOINT
        self.timeout = timeout or config.JEV_TIMEOUT_SECONDS
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.JEV_CONFIDENCE_THRESHOLD
        )

    def _request_choice(self, query):
        """向官方 System One 接口提交一个二选一问题，并返回其结构化答案。"""

        if not self.api_key:
            raise RuntimeError("未配置 JEV_API_KEY")

        payload = {
            "model": self.model,
            "state": {"user_question": query},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": (
                        "判断用户问题应该进入专业知识检索增强生成，还是直接由通用大模型回答。"
                        "只按问题内容判断，不要因为问题中偶然出现一个技术词就判为专业问题。"
                    ),
                    "criteria": {
                        self.PROFESSIONAL: (
                            "问题实质涉及 RAG、Agent、人工智能、机器学习、大模型、检索、提示词、"
                            "向量数据库或相关系统的设计、原理、开发、调试和工程实践。"
                        ),
                        self.GENERAL: (
                            "问题不属于上述 AI/Agent/RAG 专业领域，例如日常闲聊、通用常识、"
                            "普通计算，以及与技术无关的写作或生活问题。"
                        ),
                    },
                }
            },
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Jev 路由请求失败：{error}") from error

        answer = result.get("answers", {}).get("route", {})
        choice = answer.get("choice")
        confidence = float(answer.get("confidence", 0.0))
        return choice, confidence

    def predict_category(self, query):
        """返回 RAGSystem 兼容的分类标签；低置信度或异常时选择专业 RAG。"""

        try:
            choice, confidence = self._request_choice(query)
        except Exception as error:
            logger.warning("Jev 分类不可用，问题安全地进入完整 RAG：%s", error)
            return "专业咨询"

        if confidence < self.confidence_threshold:
            logger.info(
                "Jev 路由置信度不足，使用完整 RAG：choice=%s, confidence=%.3f",
                choice,
                confidence,
            )
            return "专业咨询"

        logger.info(
            "Jev 完成问题路由：choice=%s, confidence=%.3f, threshold=%.3f",
            choice,
            confidence,
            self.confidence_threshold,
        )
        if choice == self.GENERAL:
            return "通用知识"
        if choice == self.PROFESSIONAL:
            return "专业咨询"

        logger.warning("Jev 返回未知分类选项 %r，问题安全地进入完整 RAG", choice)
        return "专业咨询"
