"""
使用 jieba 对中文问句进行小写化与分词，输出 BM25 建索引和查询时使用的词项列表。
建立索引和搜索必须调用相同的预处理规则，否则词项不一致会降低匹配质量。
它只服务于词面检索，不计算句向量相似度。
"""

import jieba

from base.logger import logger

def preprocess_text(text):
    
    logger.info("开始预处理文本")
    try:
        
        return jieba.lcut(text.lower())
    except AttributeError as e:
        
        logger.error(f"文本预处理失败: {e}")
        
        return []

if __name__ == '__main__':
    
    print(preprocess_text("黑马程序员"))
