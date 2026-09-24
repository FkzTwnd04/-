"""配置项目统一日志，输出到控制台和带大小轮转的文件。

所有模块共享同一个 logger；日志格式会自动附加当前请求的 trace_id，便于按请求排查流程。
日志文件路径来自统一配置，不在业务文件中重复拼接或创建日志处理器。
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from base.config import config
from base.tracing import current_trace_id


class TraceIdFilter(logging.Filter):
    """给每条日志补上当前异步上下文的追踪编号，非请求日志使用短横线。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = current_trace_id()
        return True


def setup_logger(log_file: str = None) -> logging.Logger:
    """创建统一日志记录器；log_file 可用于部署时覆盖默认日志文件路径。"""
    target_path = os.path.abspath(log_file or config.LOG_FILE)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    project_logger = logging.getLogger("ZhiXueDa")
    project_logger.setLevel(logging.INFO)
    project_logger.propagate = False

    if not project_logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] [trace_id=%(trace_id)s] %(message)s"
        )
        trace_filter = TraceIdFilter()

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(trace_filter)

        file_handler = RotatingFileHandler(
            target_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(trace_filter)

        project_logger.addHandler(console_handler)
        project_logger.addHandler(file_handler)

    return project_logger


logger = setup_logger()
