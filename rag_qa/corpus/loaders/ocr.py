"""
为各格式加载器提供共享的图片文字识别能力：输入图片文件、图片字节或像素数组，输出识别出的文本。
OCR 引擎按需创建并在进程内复用；它不决定切分粒度、不保存文档记录，也不访问向量库。
模糊或复杂版面的图片识别效果有限，重要资料入库后应抽查识别结果。
"""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

_engine = None


def get_ocr_engine():
    """
    函数功能: 返回进程内共享的 RapidOCR 引擎; 装了 rapidocr_paddle 时用 GPU 版本, 否则使用 CPU 上的 rapidocr_onnxruntime.
    :return: 可直接调用的 RapidOCR 实例, 调用结果为 (识别结果列表, 耗时)
    """
    global _engine
    if _engine is None:
        try:
            from rapidocr_paddle import RapidOCR
            _engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR
            _engine = RapidOCR()
    return _engine


def _to_array(image) -> np.ndarray:
    """
    函数功能: 把图片路径、字节、PIL 图片统一转换成 RGB 像素数组, 已是数组时原样返回.
    :param image: 图片路径(str/Path)、图片字节、PIL.Image 或 numpy 数组
    :return: numpy 像素数组
    """
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (str, Path)):
        image = Path(image).read_bytes()
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(BytesIO(image))
    return np.array(image.convert("RGB"))


def recognize_text(image) -> str:
    """
    函数功能: 识别一张图片中的文字, 按识别顺序每行一段拼接.
    :param image: 图片路径、图片字节、PIL.Image 或 numpy 像素数组
    :return: 识别出的文本; 没有文字时返回空字符串
    """
    result, _ = get_ocr_engine()(_to_array(image))
    return "\n".join(item[1] for item in result or [])
