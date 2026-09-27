"""
逐页读取 PPTX 幻灯片，按从上到下、从左到右的阅读顺序提取文本框、表格和图片中的文字。
加载器不负责跨页合并策略、中文切块或 Milvus 写入；这些步骤分别由语料处理和检索模块承担。
"""

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from rag_qa.corpus.loaders.base import FileTextLoader, join_lines
from rag_qa.corpus.loaders.ocr import recognize_text


def _reading_order(shape):
    """
    函数功能: 生成形状的排序键, 位置缺失的形状排在最前.
    :param shape: python-pptx 形状对象
    :return: (上边距, 左边距) 元组
    """
    return (shape.top or 0, shape.left or 0)


class SlideLoader(FileTextLoader):

    def extract_text(self) -> str:
        """
        函数功能: 按幻灯片顺序拼接每页所有形状的文字.
        :return: 演示文稿全文字符串
        """
        presentation = Presentation(self.file_path)
        lines = []
        for slide in presentation.slides:
            for shape in sorted(slide.shapes, key=_reading_order):
                lines.extend(self._shape_lines(shape))
        return join_lines(lines)

    def _shape_lines(self, shape):
        """
        函数功能: 提取单个形状的文字; 组合形状递归展开, 表格按行输出, 图片走 OCR.
        :param shape: python-pptx 形状对象
        :return: 生成器, 产出文本行
        """
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for child in sorted(shape.shapes, key=_reading_order):
                yield from self._shape_lines(child)
            return
        if shape.has_text_frame:
            yield shape.text_frame.text
        if shape.has_table:
            for row in shape.table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    yield " | ".join(cells)
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            yield recognize_text(shape.image.blob)
