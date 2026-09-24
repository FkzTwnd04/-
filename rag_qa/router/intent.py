"""识别简单问候并返回通用的智学答话术，避免问候进入题库近似匹配。

本模块只处理问候和身份询问；其余问题由集成问答流程处理。
"""

import re

from typing import Optional

from base.logger import logger

GREETING_PATTERNS = (
    (r"^(你好|您好|hi|hello)[！!。,.？?\s]*$", "你好！我是智学答，有什么可以帮你？"),
    (r"^(你是谁|您是谁|你叫什么|你的名字|who are you)[？?！!。\s]*$", "我是智学答，可以帮你解答问题。"),
    (r"^(在吗|在不在|有人吗)[？?！!。\s]*$", "我在，有什么问题需要帮忙？"),
    (r"^(早上好|中午好|下午好|晚上好)[！!。,.？?\s]*$", "你好！有什么问题需要我帮你解答？"),
)

def check_greeting(query: str) -> Optional[str]:
    """
    函数功能: 去除首尾空白后按 GREETING_PATTERNS 顺序进行忽略大小写的正则匹配, 命中则返回对应固定回复.
    :param query: 用户查询文本
    :return: 命中的问候回复文本；未命中返回 None
    """
    query_text = query.strip()  
    for pattern, response in GREETING_PATTERNS:
        if re.match(pattern, query_text, re.IGNORECASE):
            return response
    return None

def route(query: str) -> str:
    """
    函数功能: L0 意图路由, 成本最低的降本手段.
    :param query: 用户查询文本
    :return: "chitchat" -> 简单问候；调用方直接返回通用固定话术
             "qa"       -> 其他问题，按正常问答流程处理
    """
    
    if not query or not isinstance(query, str):
        return "qa"

    try:
        
        if check_greeting(query):
            return "chitchat"
        
        return "qa"
    except Exception as e:
        
        logger.error(f"意图路由异常, 兜底走问答链路: {e}")
        return "qa"

def dispatch(query: str):
    """
    函数功能: 识别简单问候并返回对应的通用固定回复。
    :param query: 用户查询文本
    :return: (intent, response)；普通问题的 response 为 None。
    """
    
    intent = route(query)
    
    if intent == "chitchat":
        return intent, check_greeting(query)
    
    return intent, None
