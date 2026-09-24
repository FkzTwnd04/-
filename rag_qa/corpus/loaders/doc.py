"""
适配 Word 文档读取，将文件内容转换为 LangChain 文档或项目后续处理可使用的数据，
并尽量保留文件名、页码等来源元信息。
此模块只解决 Word 格式解析；
文本规范化和分块由 processor/splitters 处理，向量化与写库由 retriev
er 负责。
"""

from typing import Iterator
from rag_qa.corpus.loaders.ocr import get_ocr

from tqdm import tqdm
from docx.table import _Cell, Table  
from docx.oxml.table import CT_Tbl  
from docx.oxml.text.paragraph import CT_P  
from docx.text.paragraph import Paragraph  
from docx import Document as Docu1
from docx.document import Document as Docu2
from docx import ImagePart  
from PIL import Image  
from io import BytesIO  
import numpy as np  
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

class OCRDOCLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, filepath: str) -> None:
        """Initialize the loader with a file path.

        Args:
            filepath_path: The path to the filepath to load.
        """
        self.filepath = filepath

    def lazy_load(self) -> Iterator[Document]:
        
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        line = self.doc2text(self.filepath)
        yield Document(page_content=line, metadata={"source": self.filepath})

    def doc2text(self, filepath):

        
        ocr = get_ocr()
        

        
        doc = Docu1(filepath)
        
        
        resp = ""
        
        def iter_block_items(parent):
            
            if isinstance(parent, Docu2):
                parent_elm = parent.element.body
            
            elif isinstance(parent, _Cell):
                parent_elm = parent._tc
            else:
                raise ValueError("OCRDOCLoader parse fail")  
            
            
            
            for child in parent_elm.iterchildren():
                
                if isinstance(child, CT_P):  
                    yield Paragraph(child, parent)  
                elif isinstance(child, CT_Tbl):  
                    yield Table(child, parent)  

        
        
        
        b_unit = tqdm(total=len(doc.paragraphs) + len(doc.tables),
                      desc="OCRDOCLoader block index: 0")

        
        for i, block in enumerate(iter_block_items(doc)):
            
            b_unit.set_description("OCRDOCLoader  block index: {}".format(i))
            b_unit.refresh()  

            
            if isinstance(block, Paragraph):
                resp += block.text.strip() + "\n"  
                
                images = block._element.xpath('.//pic:pic')
                for image in images:
                    
                    for img_id in image.xpath('.//a:blip/@r:embed'):
                        part = doc.part.related_parts[img_id]  
                        if isinstance(part, ImagePart):  
                            
                            
                            image = Image.open(BytesIO(part._blob))  
                            result, _ = ocr(np.array(image))  
                            if result:  
                                ocr_result = [line[1] for line in result]  
                                resp += "\n".join(ocr_result)  
            
            elif isinstance(block, Table):
                
                for row in block.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            resp += paragraph.text.strip() + "\n"  

            
            b_unit.update(1)
        
        return resp

if __name__ == '__main__':
    docx_loader = OCRDOCLoader(filepath='/0001.项目目录/others/test/人工智能就业课课程大纲.docx')
    doc = docx_loader.load()
    print(doc)