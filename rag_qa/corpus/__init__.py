"""
汇总 corpus 子包对外使用的文档处理函数、加载器和切分器。
调用方可从语料包统一导入常用能力，而不需要知道每种文件格式对应的内部文件。
该包负责文本生成与整理，不负责文档数据库状态、HTTP 上传或 Milvus 底层连接。
"""

from rag_qa.corpus.processor import (      
    document_loaders,
    load_documents_from_directory,
    process_documents,
)

try:
    from rag_qa.corpus.splitters import ChineseRecursiveTextSplitter, AliTextSplitter
except ImportError:      
    ChineseRecursiveTextSplitter = None
    AliTextSplitter = None

try:
    from rag_qa.corpus.loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
except ImportError:      
    OCRPDFLoader = OCRDOCLoader = OCRPPTLoader = OCRIMGLoader = None

try:
    from rag_qa.corpus.dedup import content_hash, dedup_documents
except ImportError:      
    content_hash = None
    dedup_documents = None

try:
    from rag_qa.corpus.indexer import CorpusIndexer
except ImportError:      
    CorpusIndexer = None

__all__ = [
    "document_loaders",
    "load_documents_from_directory",
    "process_documents",
    "ChineseRecursiveTextSplitter",
    "AliTextSplitter",
    "OCRPDFLoader",
    "OCRDOCLoader",
    "OCRPPTLoader",
    "OCRIMGLoader",
    "content_hash",
    "dedup_documents",
    "CorpusIndexer",
]
