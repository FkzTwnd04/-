"""
作为 API 包的统一入口，导出应用工厂 create_app，供 Uvicorn 或其他启动脚
本创建 FastAPI 应用。
这里只公开接入层能力，不在导入时执行问答、打开数据库连接或加载模型。
"""

from api.server import create_app

__all__ = ["create_app"]
