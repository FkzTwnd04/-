"""
定位项目根目录下的 config.ini 与 .env，读取普通设置及密钥设置，并统一提供模型、
存储、日志和检索所需参数。
配置优先级和敏感值来源在本模块集中处理，列表等结构化值使用安全解析。
更换部署环境时应调整环境变量或配置文件，不要把真实密钥写进代码或提交到版本库。
"""

import configparser                 
import os                           
import ast                          
import warnings                     

current_file_path = os.path.abspath(__file__)

current_dir_path = os.path.dirname(current_file_path)

project_root = os.path.dirname(current_dir_path)

config_file_path = os.path.join(project_root, 'config.ini')

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
    
    def __init__(self, config_file=config_file_path):
        """
        函数功能: 读取 config.ini 并结合环境变量初始化项目路径、MySQL/Redis/Milvus、大模型、Jev、检索、意图/FAQ 阈值、上传摄取和日志等全部配置项; 密钥缺失时只发出警告不抛异常.
        :param config_file: config.ini 文件路径, 传 None 时使用项目根目录下的 config.ini
        :return: None
        """
        
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        
        self.PROJECT_ROOT = project_root

        
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        
        self.CORPUS_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/corpus/loaders')

        
        self.ENV_SOURCE = ENV_SOURCE
        self.secret_status = {}     

        
        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        self.config.read(config_file, encoding='utf-8')

        
        
        self.MYSQL_HOST = self._setting('MYSQL_HOST', 'mysql', 'host', 'localhost')
        self.MYSQL_USER = self._setting('MYSQL_USER', 'mysql', 'user', 'root')
        self.MYSQL_PASSWORD = self._secret('MYSQL_PASSWORD', 'mysql', 'password', 'MySQL密码')
        self.MYSQL_DATABASE = self._setting('MYSQL_DATABASE', 'mysql', 'database', 'subjects_kg')

        
        self.REDIS_HOST = self._setting('REDIS_HOST', 'redis', 'host', 'localhost')
        self.REDIS_PORT = int(self._setting('REDIS_PORT', 'redis', 'port', 6379))
        self.REDIS_PASSWORD = self._secret('REDIS_PASSWORD', 'redis', 'password', 'Redis密码', required=False)
        self.REDIS_DB = int(self._setting('REDIS_DB', 'redis', 'db', 0))

        
        self.MILVUS_HOST = self._setting('MILVUS_HOST', 'milvus', 'host', 'localhost')
        self.MILVUS_PORT = self._setting('MILVUS_PORT', 'milvus', 'port', '19530')
        self.MILVUS_DATABASE_NAME = self._setting('MILVUS_DATABASE_NAME', 'milvus', 'database_name', 'itcast')
        self.MILVUS_COLLECTION_NAME = self._setting('MILVUS_COLLECTION_NAME', 'milvus', 'collection_name',
                                                    'edurag_final')

        
        self.LLM_MODEL = self._setting('LLM_MODEL', 'llm', 'model', 'qwen-plus')
        self.DASHSCOPE_API_KEY = self._secret('DASHSCOPE_API_KEY', 'llm', 'dashscope_api_key', '大模型密钥')
        self.DASHSCOPE_BASE_URL = self._setting('DASHSCOPE_BASE_URL', 'llm', 'dashscope_base_url',
                                                'https://dashscope.aliyuncs.com/compatible-mode/v1')

        # Jev 仅用于在线问题路由；密钥可留空，服务不可用时路由器会安全回退到完整 RAG。
        self.JEV_API_KEY = self._secret('JEV_API_KEY', 'jev', 'api_key', 'Jev API 密钥', required=False)
        self.JEV_MODEL = self._setting('JEV_MODEL', 'jev', 'model', 'jev-latest')
        self.JEV_TIMEOUT_SECONDS = float(self._setting('JEV_TIMEOUT_SECONDS', 'jev', 'timeout_seconds', 8))
        self.JEV_CONFIDENCE_THRESHOLD = float(self._setting('JEV_CONFIDENCE_THRESHOLD', 'jev', 'confidence_threshold', 0.7))

        
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 意图检索阈值决定何时接受代表问句命中；上线前必须用独立标注集校准。
        self.INTENT_BM25_THRESHOLD = float(self._setting('INTENT_BM25_THRESHOLD', 'intent', 'bm25_threshold', 0.85))
        self.INTENT_INDEX_TTL_SECONDS = int(self._setting('INTENT_INDEX_TTL_SECONDS', 'intent', 'index_ttl_seconds', 60))
        self.HOT_QUESTION_BM25_THRESHOLD = float(self._setting('HOT_QUESTION_BM25_THRESHOLD', 'hot_questions', 'bm25_threshold', 0.85))
        self.HOT_QUESTION_REFRESH_SECONDS = int(self._setting('HOT_QUESTION_REFRESH_SECONDS', 'hot_questions', 'refresh_seconds', 60))
        self.FAQ_L2A_THRESHOLD = float(self._setting('FAQ_L2A_THRESHOLD', 'faq', 'bm25_threshold', 0.85))
        self.FAQ_L2A_MARGIN = float(self._setting('FAQ_L2A_MARGIN', 'faq', 'bm25_margin', 0.05))
        self.FAQ_L2B_THRESHOLD = float(self._setting('FAQ_L2B_THRESHOLD', 'faq', 'vector_threshold', 0.92))
        self.FAQ_L2B_MARGIN = float(self._setting('FAQ_L2B_MARGIN', 'faq', 'vector_margin', 0.05))
        self.FAQ_CACHE_TTL_SECONDS = int(self._setting('FAQ_CACHE_TTL_SECONDS', 'faq', 'cache_ttl_seconds', 3600))
        self.DOCUMENT_UPLOAD_DIR = self._setting('DOCUMENT_UPLOAD_DIR', 'ingest', 'upload_dir', 'rag_qa/data/uploads')
        self.DOCUMENT_MAX_UPLOAD_BYTES = int(self._setting('DOCUMENT_MAX_UPLOAD_BYTES', 'ingest', 'max_upload_bytes', 52428800))
        self.INGEST_MAX_CONCURRENT = int(self._setting('INGEST_MAX_CONCURRENT', 'ingest', 'max_concurrent', 2))

        
        self.VALID_SOURCES = ast.literal_eval(
            self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]'))
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='13112345678')

        
        log_filename = self.config.get('logger', 'log_file', fallback='logs/app.log')
        
        self.LOG_FILE = os.path.join(self.PROJECT_ROOT, log_filename) if '/' in log_filename \
            else os.path.join(self.LOG_DIR, log_filename)

    
    def _setting(self, env_key, section, option, fallback):
        """
        函数功能: 读取非敏感配置, 保证部署时可用环境变量覆盖 ini 配置.
        :param env_key: 环境变量名, 例如: MYSQL_HOST
        :param section: ini 节点名, 例如: mysql
        :param option: ini 配置键名, 例如: host
        :param fallback: 都没配时使用的默认值
        :return: 最终配置值(字符串)
        """
        
        value = os.getenv(env_key)
        if value is not None and value != '':
            return value
        
        return self.config.get(section, option, fallback=fallback)

    
    def _secret(self, env_key, section, option, description, required=True):
        """
        函数功能: 读取敏感配置(密钥/密码), 杜绝把明文写进版本库.
        :param env_key: 环境变量名, 例如: DASHSCOPE_API_KEY
        :param section: ini 节点名(仅用于兼容历史配置)
        :param option: ini 配置键名(仅用于兼容历史配置)
        :param description: 该配置的中文说明, 用于告警提示
        :param required: 是否必需, False 表示允许为空(例如 Redis 无密码)
        :return: 配置值(字符串), 未配置时返回空字符串
        """
        
        value = (os.getenv(env_key) or '').strip()
        if value:
            self.secret_status[env_key] = 'env'
            return value

        
        legacy_value = self.config.get(section, option, fallback='').strip()
        if legacy_value and not legacy_value.startswith('<'):
            self.secret_status[env_key] = 'config.ini(不推荐)'
            warnings.warn(
                f"敏感配置 {env_key} 仍写在 config.ini 中, 存在泄露风险, 请迁移到 .env(参考 .env.example)",
                stacklevel=2
            )
            return legacy_value

        
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
    
    config_file = config_file_path     

    
    conf = Config(config_file)
    
    print(f'env加载方式: {conf.ENV_SOURCE}')
    print(f'mysql: {conf.MYSQL_HOST}, {conf.MYSQL_USER}, 密码已配置={bool(conf.MYSQL_PASSWORD)}, {conf.MYSQL_DATABASE}')
    print(f'redis: {conf.REDIS_HOST}, {conf.REDIS_PORT}, 密码已配置={bool(conf.REDIS_PASSWORD)}, {conf.REDIS_DB}')
    print(f'密钥已配置: {bool(conf.DASHSCOPE_API_KEY)}')
    print(f'敏感配置状态: {conf.secret_status}')
    print(f'缺失的必需配置: {conf.missing_secrets()}')
    print(f'log: {conf.LOG_FILE}')
