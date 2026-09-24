"""
封装 Redis 连接和通用 JSON/答案读写操作。
本类不判断问题是否适合复用，也不负责统一生成带租户/知识库版本的键；调用方必须负责答案准入、隔离和失效策略。
FAQ 的 L1 答案缓存目前由 `storage/faq.py` 按租户与知识库版本管理。
"""

import redis
import json
import os, sys

from base.config import config
from base.logger import logger

class RedisClient:
    
    def __init__(self):
        
        self.logger = logger

        
        try:
            self.client = redis.StrictRedis(
                host=config.REDIS_HOST,           
                port=config.REDIS_PORT,           
                password=config.REDIS_PASSWORD,   
                db=config.REDIS_DB,               
                decode_responses=True               
            )
        except redis.RedisError as e:
            self.logger.error(f'Redis连接异常: {e}')
            raise   

        
        self.logger.info('Redis连接成功')

    
    def set_data(self, key, value):
        try:
            
            self.client.set(key, json.dumps(value, ensure_ascii=False)) 
            
            self.logger.info(f'Redis存储数据成功: {key}')
        except redis.RedisError as e:
            self.logger.error(f'Redis存储数据失败: {e}')

    
    def get_data(self, key):
        try:
            
            data = self.client.get(key)
            
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'Redis获取数据失败: {e}')
            return None     

    
    def get_answer(self, query):
        try:
            
            answer = self.client.get(f"answer:{query}")
            
            if answer:
                self.logger.info(f'从Redis中获取答案成功: {query}')
                return answer
            
            return None
        except redis.RedisError as e:
            self.logger.error(f'从Redis中获取答案失败: {e}')
            return None

if __name__ == '__main__':
    
    redCli = RedisClient()

    
    print(redCli.client.keys("*"))

    
    

    
    
