"""提供 HTTP 请求追踪编号和轻量阶段计时能力。

trace_id 使用异步上下文存储，确保并发请求读取到各自的编号；中间件应在请求结束时恢复上下文。
StageTimer 供业务阶段包围调用并读取耗时。本模块不自动统计模型 token 或推理费用。
"""

from contextvars import ContextVar
from time import perf_counter
from uuid import uuid4


_trace_id = ContextVar("trace_id", default="-")


def current_trace_id() -> str:
    """返回当前异步请求的追踪编号；不在请求上下文时返回短横线。"""
    return _trace_id.get()


def set_trace_id(value: str = ""):
    """设置或生成当前上下文的编号，并返回可用于恢复上下文的 token。"""
    return _trace_id.set(value or uuid4().hex)


def reset_trace_id(token) -> None:
    """恢复中间件进入前的上下文，防止同一进程不同请求串号。"""
    _trace_id.reset(token)


class StageTimer:
    """记录一段同步阶段耗时；elapsed_ms 在 with 结束后可读取。"""

    def __init__(self):
        self.elapsed_ms = 0.0
        self._started = 0.0

    def __enter__(self):
        self._started = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed_ms = (perf_counter() - self._started) * 1000
        return False
