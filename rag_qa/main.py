"""
提供本地命令行入口，按选择执行文档目录处理或交互式 RAG 提问。
文档模式调用 corpus processor 并将处理结果交给 HybridRetriever
 写入；
查询模式构造 RAGSystem 并处理用户输入。
它适用于开发和手动操作，不替代 api/server.py 的 Web 服务入口。
"""

import os
import sys
from base.config import config as conf
from base.logger import logger
from rag_qa.corpus.processor import process_documents  
from rag_qa.retriever.hybrid import HybridRetriever    
from rag_qa.generation.llm import LLMClient                   
from rag_qa.pipeline.qa import RAGSystem                      

def main(query_mode=True, directory_path="data"):
    """
    函数功能: 系统主函数, 控制两种运行模式.
    :param query_mode:
        数据处理模式: query_model=False  解析指定目录的文档, 分割为文档块并存入向量数据.
        交互式查询模式: query_model=True 接收用户问题, 调用RAG系统生成答案并展示.
    :param directory_path: 数据处理模式下的文档根目录, 默认是: ./data, 需包含各个学科的子目录, 例如: ai_data, java_data
    :return: 无
    """

    
    try:
        llm = LLMClient()
    except Exception as e:
        logger.error(f"初始化大模型客户端失败 (请检查 API Key 和 Base URL): {e}")
        
        if query_mode: 
             print("错误：无法初始化语言模型客户端，无法进入查询模式。")
             return
        
        llm = None 

    
    
    
    
    try:
        vector_store = HybridRetriever()
    except Exception as e:
        logger.error(f"初始化 VectorStore 失败 (请检查 Milvus 连接配置): {e}")
        print("错误：无法连接到向量数据库，程序无法继续。")
        return

    
    if not query_mode:
        
        logger.info("进入数据处理模式...")
        total_chunks_added = 0
        for source_dir in conf.VALID_SOURCES:
            dir_path = os.path.join(directory_path, f"{source_dir}_data")
            if os.path.exists(dir_path):
                logger.info(f"开始处理目录: {dir_path}")
                try:
                    chunks = process_documents(
                        dir_path,
                        conf.PARENT_CHUNK_SIZE,
                        conf.CHILD_CHUNK_SIZE,
                        conf.CHUNK_OVERLAP,
                    )
                    if chunks:
                        vector_store.add(chunks)
                        total_chunks_added += len(chunks)
                        logger.info(f"成功处理目录 {dir_path}，添加了 {len(chunks)} 个文档块")
                    else:
                        logger.info(f"目录 {dir_path} 未发现有效文档或处理结果为空")
                except Exception as e:
                    logger.error(f"处理目录 {dir_path} 时出错: {e}")
            else:
                logger.warning(f"目录 {dir_path} 不存在，跳过处理")
        logger.info(f"数据处理完成，共添加了 {total_chunks_added} 个文档块到向量存储")
    else:
        
        if not llm: 
            print("错误：查询模式需要语言模型客户端，但初始化失败。")
            return

        logger.info("进入交互式查询模式...")
        try:
            rag_system = RAGSystem(vector_store, llm)
        except Exception as e:
             logger.error(f"初始化 RAGSystem 失败: {e}")
             print("错误：无法初始化 RAG 系统，无法进入查询模式。")
             return

        valid_sources = conf.VALID_SOURCES
        
        print("\n欢迎使用企业知识问答系统！")
        print(f"支持的文档来源范围：{valid_sources}")
        print("输入您的问题，或输入 'exit' 退出。")

        while True:
            query = input("\n请输入您的问题：")
            if query.lower() == "exit":
                logger.info("用户退出查询模式")
                print("再见！")
                break

            source_filter_input = input(f"请输入文档来源 ({'/'.join(valid_sources)}) (直接回车默认不过滤)：").strip()
            source_filter = None 
            if source_filter_input:
                if source_filter_input in valid_sources:
                    source_filter = source_filter_input
                    logger.info(f"用户选择了学科过滤: {source_filter}")
                else:
                    logger.warning(
                        f"无效的学科类别 '{source_filter_input}'，将不过滤"
                    )
                    print(f"提示：输入的学科 '{source_filter_input}' 无效，将不过滤。")

            try:
                print("正在生成答案，请稍候...")
                answer = rag_system.generate_answer(query, source_filter=source_filter)
                print("-" * 30)
                print(f"问题: {query}")
                print(f"回答: {answer}")
                print("-" * 30)
            except Exception as e:
                logger.error(f"处理查询 '{query}' 时失败: {str(e)}")
                print(f"抱歉，处理您的问题时遇到了错误，请稍后重试或联系管理员。\n")

if __name__ == "__main__":
    

    
    import argparse
    
    parser = argparse.ArgumentParser(description="EduRAG System Main Entry Point")
    
    parser.add_argument('--data-processing', action='store_true', help='Run in data processing mode instead of query mode.')
    
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to the data directory.')
    
    args = parser.parse_args()
    
    main(query_mode=(not args.data_processing), directory_path=args.data_dir)