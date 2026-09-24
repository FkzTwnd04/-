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
        try:
            
            self.cursor.execute("select question from jpkb")
            
            results = self.cursor.fetchall()
            
            self.logger.info('数据查询成功')
            
            return results

        except pymysql.MySQLError as e:
            
            self.logger.error(f'数据查询失败: {e}')
            
            return []

    
    def fetch_answer(self, question):
        try:
            
            self.cursor.execute("select answer from jpkb where question=%s", (question,))       
            
            result = self.cursor.fetchone()         
            
            
            
            return result[0] if result else None
        except pymysql.MySQLError as e:
            
            self.logger.error(f'答案获取失败: {e}')
            
            return None

    
    def close(self):
        
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
