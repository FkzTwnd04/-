"""
读取 PPT/PPTX 页面中的文字和来源信息，并将页面内容整理为统一文档对象交给 proces
sor。
加载器不负责跨页合并策略、中文切块或 Milvus 写入；
这些步骤分别由语料处理和检索模块承担。
"""

from typing import Iterator
from rag_qa.corpus.loaders.ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from pptx import Presentation
from PIL import Image
import numpy as np
from io import BytesIO
from tqdm import tqdm

class OCRPPTLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, filepath: str) -> None:
        """Initialize the loader with a file path.

        Args:
            filepath: The path to the ppt to load.
        """
        self.filepath = filepath

    def lazy_load(self) -> Iterator[Document]:
        
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        line = self.ppt2text(self.filepath)
        yield Document(page_content=line, metadata={"source": self.filepath})

    def ppt2text(self, filepath):
        
        prs = Presentation(filepath)
        print(f'prs-->{prs}')
        
        ocr = get_ocr()
        
        resp = ""

        def extract_text(shape):
            
            
            nonlocal resp

            
            if shape.has_text_frame:
                
                resp += shape.text.strip() + "\n"

            
            if shape.has_table:
                
                for row in shape.table.rows:
                    
                    for cell in row.cells:
                        
                        for paragraph in cell.text_frame.paragraphs:
                            
                            resp += paragraph.text.strip() + "\n"

            
            if shape.shape_type == 13:  
                
                image = Image.open(BytesIO(shape.image.blob))
                
                result, _ = ocr(np.array(image))
                if result:  
                    
                    ocr_result = [line[1] for line in result]
                    
                    resp += "\n".join(ocr_result)

            
            elif shape.shape_type == 6:  
                
                for child_shape in shape.shapes:
                    extract_text(child_shape)

        
        b_unit = tqdm(total=len(prs.slides), desc="OCRPPTLoader slide index: 1")

        
        for slide_number, slide in enumerate(prs.slides, start=1):
            
            b_unit.set_description("OCRPPTLoader slide index: {}".format(slide_number))
            b_unit.refresh()  

            
            sorted_shapes = sorted(slide.shapes, key=lambda x: (x.top, x.left))

            for shape in sorted_shapes:
                extract_text(shape)  

            b_unit.update(1)  

        return resp  

if __name__ == '__main__':
    from base.config import config
    img_loader = OCRPPTLoader(filepath=f'{config.DATA_DIR}/ai_data/01.pptx')
    doc = img_loader.load()
    print(doc)