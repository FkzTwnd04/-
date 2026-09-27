"""
统一导出 PDF、Word、PPT 和图片加载器，使 processor 可以按文件类型选择实现。
各加载器依赖不同的第三方解析库，某个库未安装时只影响对应格式，其余加载器仍可导入。
"""

from rag_qa.corpus.loaders.base import FileTextLoader

try:
    from rag_qa.corpus.loaders.pdf import PdfLoader
except ImportError:
    PdfLoader = None

try:
    from rag_qa.corpus.loaders.doc import WordLoader
except ImportError:
    WordLoader = None

try:
    from rag_qa.corpus.loaders.ppt import SlideLoader
except ImportError:
    SlideLoader = None

try:
    from rag_qa.corpus.loaders.image import ImageLoader
except ImportError:
    ImageLoader = None

__all__ = ["FileTextLoader", "PdfLoader", "WordLoader", "SlideLoader", "ImageLoader"]
