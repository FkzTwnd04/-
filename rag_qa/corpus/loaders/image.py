"""
读取图片文件并调用 OCR 提取其中的文字，供后续清洗、切分和入库。
识别质量依赖图片清晰度和版面；图片表格、多栏排版或低分辨率内容应在摄取结果中检查。
"""

from rag_qa.corpus.loaders.base import FileTextLoader
from rag_qa.corpus.loaders.ocr import recognize_text


class ImageLoader(FileTextLoader):

    def extract_text(self) -> str:
        """
        函数功能: 识别整张图片中的文字.
        :return: 识别出的文本; 没有文字时返回空字符串
        """
        return recognize_text(self.file_path)
