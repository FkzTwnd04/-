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
        """
        函数功能: 按统一配置创建 Redis 客户端(decode_responses=True), 创建时抛出 RedisError 则记录错误并重新抛出.
        :return: None
        """
        
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
        """
        函数功能: 将 value 序列化为 JSON(保留中文)后写入 Redis 指定键, 不设置过期时间; Redis 异常只记录日志不抛出.
        :param key: Redis 键名
        :param value: 可被 json.dumps 序列化的待存数据
        :return: None
        """
        try:
            
            self.client.set(key, json.dumps(value, ensure_ascii=False)) 
            
            self.logger.info(f'Redis存储数据成功: {key}')
        except redis.RedisError as e:
            self.logger.error(f'Redis存储数据失败: {e}')

    
    def get_data(self, key):
        """
        函数功能: 读取 Redis 指定键并按 JSON 反序列化; 键不存在或值为空时返回 None, Redis 异常时记录日志并返回 None.
        :param key: Redis 键名
        :return: 反序列化后的数据, 未命中或读取失败时为 None
        """
        try:
            
            data = self.client.get(key)
            
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'Redis获取数据失败: {e}')
            return None     

    
    def get_answer(self, query):
        """
        函数功能: 从 Redis 键 "answer:{query}" 读取缓存答案原文(不做 JSON 解析); 未命中或 Redis 异常时返回 None.
        :param query: 用户问题文本, 直接拼接进缓存键
        :return: 缓存的答案字符串, 未命中或读取失败时为 None
        """
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

    
    

    
    
