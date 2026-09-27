"""
按原文顺序读取 Word(.docx) 的段落和表格，段落内嵌图片交给 OCR 识别。
表格每行输出为“单元格 | 单元格”一行，保留行内对应关系。
此模块只解决 Word 格式解析；文本规范化和分块由 processor/splitters 处理。
"""

import docx
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.parts.image import ImagePart
from docx.table import Table
from docx.text.paragraph import Paragraph

from rag_qa.corpus.loaders.base import FileTextLoader, join_lines
from rag_qa.corpus.loaders.ocr import recognize_text


class WordLoader(FileTextLoader):

    def extract_text(self) -> str:
        """
        函数功能: 遍历文档主体中的段落与表格, 拼接文字、表格行和图片识别结果.
        :return: Word 全文字符串
        """
        document = docx.Document(self.file_path)
        lines = []
        for element in document.element.body.iterchildren():
            if isinstance(element, CT_P):
                lines.append(Paragraph(element, document).text)
                lines.extend(self._inline_image_text(document, element))
            elif isinstance(element, CT_Tbl):
                lines.extend(self._table_lines(Table(element, document)))
        return join_lines(lines)

    @staticmethod
    def _inline_image_text(document, paragraph_element):
        """
        函数功能: 识别段落中嵌入的图片文字.
        :param document: python-docx 文档对象
        :param paragraph_element: 段落 XML 元素
        :return: 生成器, 每张图片产出一段识别文本
        """
        for rel_id in paragraph_element.xpath(".//a:blip/@r:embed"):
            part = document.part.related_parts.get(rel_id)
            if isinstance(part, ImagePart):
                yield recognize_text(part.blob)

    @staticmethod
    def _table_lines(table):
        """
        函数功能: 把表格每一行转成一行文本, 合并单元格造成的相邻重复内容只保留一次.
        :param table: python-docx 表格对象
        :return: 生成器, 每行表格产出一行文本
        """
        for row in table.rows:
            cells = []
            for cell in row.cells:
                text = cell.text.strip()
                if text and (not cells or cells[-1] != text):
                    cells.append(text)
            if cells:
                yield " | ".join(cells)
