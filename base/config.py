"""
定位项目根目录下的 .env，读取普通设置及密钥设置，并统一提供模型、存储、日志和检索所需参数。
全部配置只来自系统环境变量和 .env；未配置的普通项使用本模块中的默认值。
.env 保存本机真实配置且不提交到版本库，可提交的配置模板是 .env.example。
"""

import ast
import json
import os
import warnings

current_file_path = os.path.abspath(__file__)

current_dir_path = os.path.dirname(current_file_path)

project_root = os.path.dirname(current_dir_path)

env_file_path = os.path.join(project_root, '.env')

def _load_env_file(path):
    """
    函数功能: 极简版 .env 解析器, 在没装 python-dotenv 时兜底使用.
    :param path: .env 文件路径
    :return: 本次成功注入的键名列表
    """
    loaded_keys = []
    
    if not os.path.exists(path):
        return loaded_keys
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            
            if key and os.getenv(key) is None:
                os.environ[key] = value
                loaded_keys.append(key)
    return loaded_keys

def _load_environment(path=env_file_path):
    """
    函数功能: 加载 .env 到环境变量, 优先使用 python-dotenv, 未安装时用内置解析器兜底.
    :param path: .env 文件路径
    :return: 字符串, 记录本次实际使用的加载方式(便于排查问题)
    """
    try:
        from dotenv import load_dotenv
        
        load_dotenv(path, override=False)
        return 'python-dotenv'
    except ImportError:
        _load_env_file(path)
        return '内置解析器(建议 pip install python-dotenv)'

ENV_SOURCE = _load_environment()

class Config:
    
    def __init__(self):
        """
        函数功能: 从环境变量(含 .env)初始化项目路径、MySQL/Redis/Milvus、大模型、Jev、检索、意图/FAQ 阈值、上传摄取和日志等全部配置项; 密钥缺失时只发出警告不抛异常.
        :return: None
        """
        self.PROJECT_ROOT = project_root

        
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        
        self.CORPUS_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/corpus/loaders')

        
        self.ENV_SOURCE = ENV_SOURCE
        self.secret_status = {}     

        
        self.MYSQL_HOST = self._setting('MYSQL_HOST', 'localhost')
        self.MYSQL_USER = self._setting('MYSQL_USER', 'root')
        self.MYSQL_PASSWORD = self._secret('MYSQL_PASSWORD', 'MySQL密码')
        self.MYSQL_DATABASE = self._setting('MYSQL_DATABASE', 'subjects_kg')

        
        self.REDIS_HOST = self._setting('REDIS_HOST', 'localhost')
        self.REDIS_PORT = int(self._setting('REDIS_PORT', 6379))
        self.REDIS_PASSWORD = self._secret('REDIS_PASSWORD', 'Redis密码', required=False)
        self.REDIS_DB = int(self._setting('REDIS_DB', 0))

        
        self.MILVUS_HOST = self._setting('MILVUS_HOST', 'localhost')
        self.MILVUS_PORT = self._setting('MILVUS_PORT', '19530')
        self.MILVUS_DATABASE_NAME = self._setting('MILVUS_DATABASE_NAME', 'itcast')
        self.MILVUS_COLLECTION_NAME = self._setting('MILVUS_COLLECTION_NAME', 'edurag_final')

        
        self.LLM_MODEL = self._setting('LLM_MODEL', 'qwen-plus')
        self.DASHSCOPE_API_KEY = self._secret('DASHSCOPE_API_KEY', '大模型密钥')
        self.DASHSCOPE_BASE_URL = self._setting('DASHSCOPE_BASE_URL',
                                                'https://dashscope.aliyuncs.com/compatible-mode/v1')

        # Jev 仅用于在线问题路由；密钥可留空，服务不可用时路由器会安全回退到完整 RAG。
        self.JEV_API_KEY = self._secret('JEV_API_KEY', 'Jev API 密钥', required=False)
        self.JEV_MODEL = self._setting('JEV_MODEL', 'typesafe/jev-1.13')
        self.JEV_ENDPOINT = self._setting('JEV_ENDPOINT', 'https://openrouter.ai/api/v1/systemone')
        self.JEV_TIMEOUT_SECONDS = float(self._setting('JEV_TIMEOUT_SECONDS', 8))
        self.JEV_CONFIDENCE_THRESHOLD = float(self._setting('JEV_CONFIDENCE_THRESHOLD', 0.7))

        
        self.PARENT_CHUNK_SIZE = int(self._setting('PARENT_CHUNK_SIZE', 1200))
        self.CHILD_CHUNK_SIZE = int(self._setting('CHILD_CHUNK_SIZE', 300))
        self.CHUNK_OVERLAP = int(self._setting('CHUNK_OVERLAP', 50))
        self.RETRIEVAL_K = int(self._setting('RETRIEVAL_K', 3))
        self.CANDIDATE_M = int(self._setting('CANDIDATE_M', 2))

        # 意图检索阈值决定何时接受代表问句命中；上线前必须用独立标注集校准。
        self.INTENT_BM25_THRESHOLD = float(self._setting('INTENT_BM25_THRESHOLD', 0.85))
        self.INTENT_INDEX_TTL_SECONDS = int(self._setting('INTENT_INDEX_TTL_SECONDS', 60))
        self.HOT_QUESTION_BM25_THRESHOLD = float(self._setting('HOT_QUESTION_BM25_THRESHOLD', 0.85))
        self.HOT_QUESTION_REFRESH_SECONDS = int(self._setting('HOT_QUESTION_REFRESH_SECONDS', 60))
        self.FAQ_L2A_THRESHOLD = float(self._setting('FAQ_L2A_THRESHOLD', 0.85))
        self.FAQ_L2A_MARGIN = float(self._setting('FAQ_L2A_MARGIN', 0.05))
        self.FAQ_L2B_THRESHOLD = float(self._setting('FAQ_L2B_THRESHOLD', 0.92))
        self.FAQ_L2B_MARGIN = float(self._setting('FAQ_L2B_MARGIN', 0.05))
        self.FAQ_CACHE_TTL_SECONDS = int(self._setting('FAQ_CACHE_TTL_SECONDS', 3600))
        self.DOCUMENT_UPLOAD_DIR = self._setting('DOCUMENT_UPLOAD_DIR', 'rag_qa/data/uploads')
        self.DOCUMENT_MAX_UPLOAD_BYTES = int(self._setting('DOCUMENT_MAX_UPLOAD_BYTES', 52428800))
        self.INGEST_MAX_CONCURRENT = int(self._setting('INGEST_MAX_CONCURRENT', 2))

        
        self.VALID_SOURCES = self._list_setting('VALID_SOURCES', ["ai", "java", "test", "ops", "bigdata"])
        self.CUSTOMER_SERVICE_PHONE = self._setting('CUSTOMER_SERVICE_PHONE', '12345678')

        
        log_filename = self._setting('LOG_FILE', 'logs/app.log')
        
        self.LOG_FILE = os.path.join(self.PROJECT_ROOT, log_filename) if '/' in log_filename \
            else os.path.join(self.LOG_DIR, log_filename)

    
    @staticmethod
    def _setting(env_key, fallback):
        """
        函数功能: 读取非敏感配置, 环境变量(含 .env)未设置或为空时使用默认值.
        :param env_key: 环境变量名, 例如: MYSQL_HOST
        :param fallback: 未配置时使用的默认值
        :return: 最终配置值(环境变量为字符串, 默认值保持原类型)
        """
        value = os.getenv(env_key)
        if value is not None and value.strip() != '':
            return value.strip()
        return fallback

    
    def _list_setting(self, env_key, fallback):
        """
        函数功能: 读取列表型配置, 支持 JSON/Python 列表写法和逗号分隔写法.
        :param env_key: 环境变量名, 例如: VALID_SOURCES
        :param fallback: 未配置时使用的默认列表
        :return: 字符串列表
        """
        raw = self._setting(env_key, None)
        if raw is None:
            return list(fallback)
        if raw.startswith('['):
            try:
                return [str(item) for item in json.loads(raw)]
            except ValueError:
                return [str(item) for item in ast.literal_eval(raw)]
        return [item.strip() for item in raw.split(',') if item.strip()]

    
    def _secret(self, env_key, description, required=True):
        """
        函数功能: 读取敏感配置(密钥/密码), 只从环境变量(含 .env)获取, 杜绝把明文写进版本库.
        :param env_key: 环境变量名, 例如: DASHSCOPE_API_KEY
        :param description: 该配置的中文说明, 用于告警提示
        :param required: 是否必需, False 表示允许为空(例如 Redis 无密码)
        :return: 配置值(字符串), 未配置时返回空字符串
        """
        value = (os.getenv(env_key) or '').strip()
        if value:
            self.secret_status[env_key] = 'env'
            return value

        
        self.secret_status[env_key] = '缺失'
        if required:
            warnings.warn(
                f"未读取到敏感配置 {env_key}({description}), 请在 {env_file_path} 中配置, 可参考 .env.example",
                stacklevel=2
            )
        return ''

    
    def missing_secrets(self):
        """
        函数功能: 列出缺失的必需敏感配置.
        :return: 列表, 例如: ['DASHSCOPE_API_KEY']
        """
        return [key for key, status in self.secret_status.items() if status == '缺失']

config = Config()

if __name__ == '__main__':
    conf = Config()
    
    print(f'env加载方式: {conf.ENV_SOURCE}')
    print(f'mysql: {conf.MYSQL_HOST}, {conf.MYSQL_USER}, 密码已配置={bool(conf.MYSQL_PASSWORD)}, {conf.MYSQL_DATABASE}')
    print(f'redis: {conf.REDIS_HOST}, {conf.REDIS_PORT}, 密码已配置={bool(conf.REDIS_PASSWORD)}, {conf.REDIS_DB}')
    print(f'密钥已配置: {bool(conf.DASHSCOPE_API_KEY)}')
    print(f'敏感配置状态: {conf.secret_status}')
    print(f'缺失的必需配置: {conf.missing_secrets()}')
    print(f'log: {conf.LOG_FILE}')
