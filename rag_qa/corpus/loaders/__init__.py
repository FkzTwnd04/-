"""
统一组织 PDF、Word、PPT 和图片加载器的导入，使 processor 可以按文件类型调
用对应实现。
OCR 是多个格式加载器可能共用的文字提取能力；
依赖缺失或文件损坏时应让调用方得到可诊断的异常，而不是误把空文本当作成功入库。
"""

from base.config import config

try:
    from .doc import *
except ImportError:
    pass

try:
    from .ppt import *
except ImportError:
    pass

try:
    from .image import *
except ImportError:
    pass

try:
    from .pdf import *
except ImportError:
    pass
