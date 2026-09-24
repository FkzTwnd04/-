"""
从统一配置创建 MySQL 连接，并提供存储模块所需的连接、游标和基础数据库访问能力。
会话及 FAQ 等组件依赖该客户端。
当前共享连接方式在多线程并发下需谨慎，连接池、事务边界、断线重连和游标安全需要单独加固。
"""

import pymysql              
import pandas as pd         
import sys, os              

from base.config import config
from base.logger import logger

class MySQLClient:
    
    def __init__(self):
        """
        函数功能: 按统一配置连接 MySQL 并创建游标; 连接失败时记录错误并重新抛出 pymysql.MySQLError.
        :return: None
        """
        
        self.logger = logger
        try:
            
            self.connection = pymysql.connect(
                host = config.MYSQL_HOST,             
                user = config.MYSQL_USER,             
                password = config.MYSQL_PASSWORD,     
                database = config.MYSQL_DATABASE      
            )
            
            self.cursor = self.connection.cursor()

            
            self.logger.info('MySQL 连接成功')
        except pymysql.MySQLError as e:
            
            self.logger.error(f'MySQL 连接失败: {e}')
            raise       

    
    def create_table(self):
        """
        函数功能: 创建 jpkb 学科问答表(不存在时才创建)并提交; 失败时记录错误并重新抛出 pymysql.MySQLError.
        :return: None
        """
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS jpkb (
            id INT AUTO_INCREMENT PRIMARY KEY,      # 自增主键
            subject_name VARCHAR(20),               # 学科名称
            question VARCHAR(1000),                 # 问题内容
            answer VARCHAR(1000))                   # 答案内容
        """

        try:
             
             self.cursor.execute(create_table_query)
             
             self.connection.commit()
             
             self.logger.info('表创建成功')
        except pymysql.MySQLError as e:
            
            self.logger.error(f'创建表失败: {e}')
            raise

    
    def insert_data(self, csv_path):
        """
        函数功能: 用 pandas 读取 CSV, 将每行的“学科名称/问题/答案”逐条插入 jpkb 表后统一提交; 任何异常都会记录日志、回滚并重新抛出.
        :param csv_path: CSV 文件路径, 需包含 学科名称、问题、答案 三列
        :return: None
        """
        try:
            
            data = pd.read_csv(csv_path)
            
            

            
            for _, row in data.iterrows():
                
                insert_query = "insert into jpkb(subject_name, question, answer) values(%s, %s, %s)"
                
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))

            
            self.connection.commit()
            
            self.logger.info('MySQL数据插入成功')

        except Exception as e:
            
            
            self.logger.error(f'数据插入失败: {e}')
            
            self.connection.rollback()
            
            raise

    
    def fetch_questions(self):
        """
        函数功能: 查询 jpkb 表中的全部问题; 查询失败时记录日志并返回空列表.
        :return: 游标 fetchall 结果, 每个元素是只含 question 的元组; 失败时为 []
        """
        try:
            
            self.cursor.execute("select question from jpkb")
            
            results = self.cursor.fetchall()
            
            self.logger.info('数据查询成功')
            
            return results

        except pymysql.MySQLError as e:
            
            self.logger.error(f'数据查询失败: {e}')
            
            return []

    
    def fetch_answer(self, question):
        """
        函数功能: 按问题文本精确匹配 jpkb 表并取第一条答案; 查询失败时记录日志并返回 None.
        :param question: 要查找的问题文本(需与表中 question 完全相等)
        :return: 答案字符串, 未找到或查询失败时为 None
        """
        try:
            
            self.cursor.execute("select answer from jpkb where question=%s", (question,))       
            
            result = self.cursor.fetchone()         
            
            
            
            return result[0] if result else None
        except pymysql.MySQLError as e:
            
            self.logger.error(f'答案获取失败: {e}')
            
            return None

    
    def close(self):
        """
        函数功能: 关闭 MySQL 连接; 关闭失败时只记录错误日志不抛出.
        :return: None
        """
        
        try:
            
            self.connection.close()
            
            self.logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            
            self.logger.error(f"关闭连接失败: {e}")

if __name__ == '__main__':
    
    mysql_client = MySQLClient()

    
    

    
    

    
    
    

    
    answer = mysql_client.fetch_answer('什么是cart算法')
    print(answer)

    
    mysql_client.close()
