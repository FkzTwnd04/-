"""
逐页提取 PDF 的文字层；页面中占比足够大的图片（通常是扫描页或截图）再交给 OCR 补充识别。
提取文本作为 processor 的输入，后续由切分器形成检索块。
PDF 解析成功不保证 OCR 结果无误，需对重要资料抽查。
"""

import fitz
import numpy as np

from rag_qa.corpus.loaders.base import FileTextLoader, join_lines
from rag_qa.corpus.loaders.ocr import recognize_text

# 图片宽、高都至少占页面这个比例才做 OCR，避免把图标、页眉装饰图也识别一遍。
MIN_IMAGE_RATIO = 0.6


class PdfLoader(FileTextLoader):

    def extract_text(self) -> str:
        """
        函数功能: 按页顺序拼接文字层与大图的 OCR 结果.
        :return: PDF 全文字符串
        """
        lines = []
        with fitz.open(self.file_path) as pdf:
            for page in pdf:
                lines.append(page.get_text("text"))
                for info in page.get_image_info(xrefs=True):
                    xref = info.get("xref")
                    if xref and self._is_large_image(info["bbox"], page.rect):
                        lines.append(recognize_text(self._image_pixels(pdf, xref, page.rotation)))
        return join_lines(lines)

    @staticmethod
    def _is_large_image(bbox, page_rect) -> bool:
        """
        函数功能: 判断图片在页面上的显示区域是否足够大.
        :param bbox: 图片显示区域 (x0, y0, x1, y1)
        :param page_rect: 页面矩形
        :return: 宽、高占比都不低于 MIN_IMAGE_RATIO 时为 True
        """
        width_ratio = (bbox[2] - bbox[0]) / page_rect.width
        height_ratio = (bbox[3] - bbox[1]) / page_rect.height
        return width_ratio >= MIN_IMAGE_RATIO and height_ratio >= MIN_IMAGE_RATIO

    @staticmethod
    def _image_pixels(pdf, xref: int, rotation: int) -> np.ndarray:
        """
        函数功能: 取出 PDF 内嵌图片并转成 RGB 像素数组, 页面有旋转时同步转正.
        :param pdf: 已打开的 fitz 文档
        :param xref: 图片对象编号
        :param rotation: 页面顺时针旋转角度(0/90/180/270)
        :return: numpy 像素数组
        """
        pix = fitz.Pixmap(pdf, xref)
        if pix.n - pix.alpha != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if rotation % 360:
            pixels = np.rot90(pixels, k=-(rotation // 90))
        return np.ascontiguousarray(pixels)
