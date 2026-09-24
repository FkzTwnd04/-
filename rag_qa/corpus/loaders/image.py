"""
读取图片文件并调用 OCR 提取其中的文字，将识别结果包装为统一文档对象，供后续清洗、切分和入库
。
识别质量依赖图片清晰度、版面和 OCR 依赖；
图片表格、多栏排版或低分辨率内容应在摄取结果中检查。
"""

from typing import Iterator
from rag_qa.corpus.loaders.ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

class OCRIMGLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, img_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            img_path: The path to the img to load.
        """
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        line = self.img2text()
        yield Document(page_content=line, metadata={"source": self.img_path})

    def img2text(self):
        resp = ""
        ocr = get_ocr()
        result, _ = ocr(self.img_path)
        if result:
            ocr_result = [line[1] for line in result]
            resp += "\n".join(ocr_result)
        return resp

if __name__ == '__main__':
    img_loader = OCRIMGLoader(img_path='/0001.项目目录/others/test/人工智能就业课课程大纲.png')
    doc = img_loader.load()
    print(doc)