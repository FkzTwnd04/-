"""
读取单个文件或目录中的文档，根据文件扩展名选择 loader，提取文本与元数据后调用切分器，生成
可交给检索层的文档块。
支持的实际格式由 document_loaders 映射及已安装依赖决定。
处理阶段关注解析与切分；
content_hash 去重、状态登记和失败重试需要后续摄取编排接入。
"""

import os
from datetime import datetime

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter

from base.config import config
from base.logger import logger
from rag_qa.corpus.splitters import ChineseRecursiveTextSplitter
from rag_qa.corpus.loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader

document_loaders = {
    ".txt": TextLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".ppt": OCRPPTLoader,
    ".pptx": OCRPPTLoader,
    ".jpg": OCRIMGLoader,
    ".png": OCRIMGLoader,
    ".md": UnstructuredMarkdownLoader,
}

def load_documents_from_directory(directory_path: str):
    """
    函数功能: 从指定目录(含子目录)加载所有支持类型的文件, 为每个文档补充元数据.
    :param directory_path: 目标文件夹路径
    :return: Document 列表, 每个元素带 source / file_path / timestamp 元数据
    """
    documents = []
    supported_extensions = document_loaders.keys()

    
    source = os.path.basename(directory_path).replace("_data", "")

    
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()

            if file_extension not in supported_extensions:
                logger.warning(f"不支持的文件类型: {file_path}")
                continue

            try:
                loader_class = document_loaders[file_extension]
                
                loader = (
                    loader_class(file_path, encoding="utf-8")
                    if file_extension == ".txt"
                    else loader_class(file_path)
                )
                loaded_docs = loader.load()

                
                for doc in loaded_docs:
                    doc.metadata["source"] = source
                    doc.metadata["file_path"] = file_path
                    doc.metadata["timestamp"] = datetime.now().isoformat()

                documents.extend(loaded_docs)
                logger.info(f"成功加载文件: {file_path}")
            except Exception as e:
                
                logger.error(f"加载文件 {file_path} 失败: {str(e)}")

    return documents

def process_documents(
    directory_path: str,
    parent_chunk_size: int = None,
    child_chunk_size: int = None,
    chunk_overlap: int = None,
):
    """
    函数功能: 加载文档并做父子分层切分, 返回子块列表(每个子块带父块关联信息).
    :param directory_path: 文档目录
    :param parent_chunk_size: 父块长度, 默认取配置
    :param child_chunk_size: 子块长度, 默认取配置
    :param chunk_overlap: 重叠长度, 默认取配置
    :return: 子块列表(langchain Document), 元数据含 parent_id / parent_content / id
    """
    parent_chunk_size = parent_chunk_size or config.PARENT_CHUNK_SIZE
    child_chunk_size = child_chunk_size or config.CHILD_CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    
    documents = load_documents_from_directory(directory_path)
    logger.info(f"加载的文档数量: {len(documents)}")

    
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    child_chunks = []

    
    for i, doc in enumerate(documents):
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()
        is_markdown = file_extension == ".md"

        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(
            f'处理文档: {doc.metadata["file_path"]}, 使用切分器: {"Markdown" if is_markdown else "ChineseRecursive"}'
        )

        
        parent_docs = parent_splitter_to_use.split_documents([doc])

        for j, parent_doc in enumerate(parent_docs):
            parent_id = f"doc_{i}_parent_{j}"
            parent_doc.metadata["parent_id"] = parent_id
            parent_doc.metadata["content"] = parent_doc.page_content

            
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])

            for k, sub_chunk in enumerate(sub_chunks):
                
                sub_chunk.metadata["parent_id"] = parent_id
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                child_chunks.append(sub_chunk)

    logger.info(f"切分完成的子块总数: {len(child_chunks)}")
    return child_chunks

if __name__ == "__main__":
    chunks = process_documents(
        "../data/ai_data", config.PARENT_CHUNK_SIZE, config.CHILD_CHUNK_SIZE, config.CHUNK_OVERLAP
    )
    print(f"切分完成的子块总数: {len(chunks)}")
    print(f"前1个子块详情: {chunks[0] if chunks else '未切分到任何子块'}")
