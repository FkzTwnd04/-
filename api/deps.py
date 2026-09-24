"""
集中管理路由执行时需要的共享依赖。
get_qa_system() 按需创建 IntegratedQASystem，并在当前进程复用
该实例；
check_secrets() 检查模型密钥等配置并记录提示。
后续可在此注入经过鉴权得到的 tenant_id，但不能从客户端任意参数直接信任租户身份。
"""

from rag_qa.pipeline.integrated import IntegratedQASystem

from base.logger import logger
from rag_qa.corpus.indexer import DocumentIndexer
from rag_qa.pipeline.ingest import IngestService
from base.config import config
from threading import Lock

_qa_system = None
_ingest_service = None
_qa_system_lock = Lock()

def get_qa_system() -> IntegratedQASystem:
    """
    函数功能: 按需创建进程内共享的 IntegratedQASystem 单例, 使用双重检查加锁保证并发下只初始化一次.
    :return: 共享的 IntegratedQASystem 实例
    """
    global _qa_system
    
    if _qa_system is None:
        with _qa_system_lock:
            if _qa_system is None:
                _qa_system = IntegratedQASystem()
    return _qa_system

def get_ingest_service() -> IngestService:
    """按需创建共享摄取服务，并复用问答系统的 MySQL 与向量检索组件。"""

    global _ingest_service
    if _ingest_service is None:
        qa_system = get_qa_system()
        indexer = DocumentIndexer(qa_system.mysql_client, qa_system.vector_store)
        _ingest_service = IngestService(indexer)
    return _ingest_service

def check_secrets() -> None:
    """检查配置密钥但不初始化数据库连接和模型，确保前端能先独立启动。"""
    missing = config.missing_secrets()
    if missing:
        logger.warning(f"以下敏感配置未读取到, 请在 .env 中配置(参考 .env.example): {missing}")
    else:
        logger.info(f"敏感配置加载完成 (来源: {config.ENV_SOURCE})")
