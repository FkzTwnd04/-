"""
解析 PDF 页面文字并保留可用的页码和文件来源；
遇到扫描页或文本抽取不足时，可通过 OCR 路径补充识别。
提取文本作为 processor 的输入，后续由切分器形成检索块。
PDF 解析成功不保证 OCR 结果无误，需对重要资料抽查。
"""

"""
OCR介绍:
    简单说就是'光学字符识别'技术, Optical Character Recognition, 能让电脑'看懂'图片里的文字,
    把图片上的手写体, 印刷体文字转换成可编辑的文本 -> 例如: 照片里的笔记,扫描版pdf的文字等.
"""

import cv2
import fitz  
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Iterator
from rag_qa.corpus.loaders.ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import CharacterTextSplitter

PDF_OCR_THRESHOLD = (0.6, 0.6)

class OCRPDFLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        line = self.pdf2text()
        yield Document(page_content=line, metadata={"source": self.file_path})

    def pdf2text(self):
        """
        函数功能: 逐页提取 PDF 文本，并对宽高占页面比例都不低于 PDF_OCR_THRESHOLD 的图片做 OCR（页面有旋转时先校正图片方向），将结果拼接为一个字符串.
        :return: 所有页面文本与图片 OCR 结果拼接后的字符串
        """
        ocr = get_ocr()
        
        doc = fitz.open(self.file_path)
        
        
        resp = ""
        b_unit = tqdm(total=doc.page_count, desc="OCRPDFLoader context page index: 0")
        for i, page in enumerate(doc):
            b_unit.set_description("OCRPDFLoader context page index: {}".format(i))
            b_unit.refresh()
            
            text = page.get_text("text")
            resp += text + "\n"
            
            
            
            img_list = page.get_image_info(xrefs=True)
            
            
            for img in img_list:
                
                if xref := img.get("xref"):
                    
                    bbox = img["bbox"]
                    
                    if ((bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                            or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                        continue
                    pix = fitz.Pixmap(doc, xref)
                    
                    if int(page.rotation) != 0:  
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                        tmp_img = Image.fromarray(img_array)
                        ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)
                        rot_img = self.rotate_img(img=ori_img, angle=360 - page.rotation)
                        img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                    else:
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

                    
                    
                    result, _ = ocr(img_array)
                    if result:
                        ocr_result = [line[1] for line in result]
                        resp += "\n".join(ocr_result)
            
            b_unit.update(1)
        return resp

    def rotate_img(self, img, angle):
        '''
        img   --image
        angle --rotation angle
        return--rotated img
        '''

        h, w = img.shape[:2]
        rotate_center = (w / 2, h / 2)
        
        
        
        
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
        return rotated_img

if __name__ == '__main__':
    from base.config import Config
    pdf_loader = OCRPDFLoader(file_path=f"{Config().DATA_DIR}/ai_data/LLM基础知识.pdf")
    doc = pdf_loader.load()

    print(type(doc))
    print(doc)
    
    
    
    