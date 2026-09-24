"""
封装对配置的大模型服务的请求，向上层提供普通文本生成或逐段流式输出。
模型名称、服务地址和密钥通过 base.config 读取；
网络异常、超时和服务端错误应由调用方明确处理。
该模块只负责模型通信，不负责检索上下文或保存会话。
"""

from typing import Iterator

from base.config import config
from base.logger import logger

class LLMClient:
    """大模型客户端: 全项目唯一的模型调用入口, 支持一次性返回 和 流式返回."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        timeout: int = 30,
        system_prompt: str = "你是一个有用的助手, 能够根据用户输入的Prompt严格执行并返回可靠的结果",
    ):
        """
        :param api_key: 大模型密钥, 默认取配置(来自 .env)
        :param base_url: 接口地址, 默认取配置
        :param model: 模型名, 默认取配置
        :param timeout: 超时秒数
        :param system_prompt: 系统提示词
        """
        from openai import OpenAI

        
        self.api_key = api_key or config.DASHSCOPE_API_KEY
        self.base_url = base_url or config.DASHSCOPE_BASE_URL
        self.model = model or config.LLM_MODEL
        self.timeout = timeout
        self.system_prompt = system_prompt

        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    
    def complete(self, prompt: str, temperature: float = 0.1) -> str:
        """
        函数功能: 调用大模型并一次性返回完整结果.
        :param prompt: 提示词
        :param temperature: 温度参数, 越低越稳定
        :return: 模型返回的文本(字符串), 失败时返回空字符串(由调用方决定降级策略)
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                timeout=self.timeout,
            )
            if completion.choices:
                return completion.choices[0].message.content or ""
            logger.warning("大模型返回空结果")
            return ""
        except Exception as e:
            logger.error(f"大模型调用失败: {e}")
            return ""

    
    def stream(self, prompt: str, temperature: float = 0.1) -> Iterator[str]:
        """
        函数功能: 调用大模型并以生成器方式逐 token 返回.
        :param prompt: 提示词
        :param temperature: 温度参数
        :return: 生成器, 逐段产出文本
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                timeout=self.timeout,
                stream=True,
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"大模型流式调用失败: {e}")
            yield f"错误: 大模型调用失败 - {e}"

    
    def __call__(self, prompt: str) -> str:
        return self.complete(prompt)

class CallableLLM:
    """
    把一个 callable(可能是返回字符串的函数, 也可能是返回生成器的流式函数) 包装成 LLMClient 那样的统一接口.
    存在的意义: 让 new_main.py 里已有的流式函数 call_dashscope 能直接接进新的分层结构, 不用改它的实现.
    """

    def __init__(self, fn):
        self.fn = fn

    def complete(self, prompt: str, temperature: float = 0.1) -> str:
        result = self.fn(prompt)
        
        if hasattr(result, "__iter__") and not isinstance(result, str):
            return "".join(str(part) for part in result)
        return result or ""

    def stream(self, prompt: str, temperature: float = 0.1) -> Iterator[str]:
        result = self.fn(prompt)
        
        if hasattr(result, "__iter__") and not isinstance(result, str):
            for part in result:
                yield str(part)
        else:
            yield str(result or "")

    def __call__(self, prompt: str) -> str:
        return self.complete(prompt)

def as_llm(llm):
    """
    函数功能: 把任意形态的模型调用对象归一成具备 complete/stream 的对象.
    :param llm: LLMClient 实例, 或 callable(返回字符串或生成器)
    :return: 具备 complete() 和 stream() 的对象
    """
    if llm is None:
        return LLMClient()
    if hasattr(llm, "complete") and hasattr(llm, "stream"):
        return llm
    if callable(llm):
        return CallableLLM(llm)
    raise TypeError(f"不支持的 llm 类型: {type(llm)}")
