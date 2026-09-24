"""
管理 MySQL 会话历史，包括建表、读取最近对话、保存问答及清理会话。
pipeline 将历史传给 RAG 提示以支持多轮交流，并可限制保留轮数来控制上下文长度。
后续增加 hit_layer 或 feedback 字段时，要同步更新表结构、写入代码和运营统计
。
"""

import pymysql

class ConversationStore:
    
    def __init__(self, mysql_client, logger):
        """
        函数功能: 保存 MySQL 客户端与日志对象, 并将每个会话保留的最大对话轮数设为 5.
        :param mysql_client: 已连接的 MySQL 客户端, 需提供 cursor 与 connection
        :param logger: 日志记录器
        :return: None
        """
        
        self.mysql_client = mysql_client
        
        self.logger = logger
        
        self.max_rounds = 5

    
    def init_conversation_table(self):
        """
        函数功能: 初始化MySQL中的 Conversations表, 用于存储对话历史.
        :return:
        """
        try:
            
            self.mysql_client.cursor.execute("""
                create table if not exists conversations(
                    id INT AUTO_INCREMENT PRIMARY KEY,          # 主键id
                    session_id VARCHAR(36) NOT NULL,            # 会话id
                    question TEXT NOT NULL,                     # 问题
                    answer TEXT NOT NULL,                       # 答案
                    timestamp DATETIME NOT NULL,                # 时间戳
                    INDEX idx_session_id (session_id)           # 创建索引列, 目的: 提高查询的效率.
                )
            """)
            
            self.mysql_client.connection.commit()
            
            self.logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            
            self.logger.error(f"初始化对话历史表失败: {e}")
            
            raise

    
    def _fetch_recent_history(self, session_id):
        """
        函数功能: 获取最近5轮对话历史
        :param session_id: 会话唯一标识(字符串), 用于定位具体会话.
        :return: 列表, 包含字典形式的对话记录, 例如: [{'question':问题, 'answer':答案}]
        """
        try:
            
            self.mysql_client.cursor.execute("""
                 SELECT question, answer
                 FROM conversations
                 WHERE session_id = %s
                 ORDER BY timestamp DESC
                     LIMIT %s
                 """, (session_id, self.max_rounds))
            
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            
            return history[::-1]
        except pymysql.MySQLError as e:
            
            self.logger.error(f"获取对话历史失败: {e}")
            
            return []

    
    def get_session_history(self, session_id):
        """
        函数功能: 从MySQL获取会话历史.
        :param session_id: 会话唯一标识(字符串)
        :return: 列表, 包含最近5轮对话记录 -> 同 _fetch_recent_history()的返回值.
        """
        
        return self._fetch_recent_history(session_id)

    
    def update_session_history(self, session_id: str, question: str, answer: str):
        """
        函数功能: 更新会话历史到MySQL，保留最近5轮对话
        :param session_id: 会话唯一标识(字符串)
        :param question: 用户的问题(字符串)
        :param answer: 系统生成的答案(字符串)
        :return:
        """
        try:
            
            self.mysql_client.cursor.execute("""
                 INSERT INTO conversations (session_id, question, answer, timestamp)
                 VALUES (%s, %s, %s, NOW())
                 """, (session_id, question, answer))
            
            history = self._fetch_recent_history(session_id)
            
            self.mysql_client.cursor.execute("""
                 DELETE
                 FROM conversations
                 WHERE session_id = %s
                   AND id NOT IN (SELECT id
                      FROM (SELECT id
                            FROM conversations
                            WHERE session_id = %s
                            ORDER BY timestamp DESC
                                LIMIT %s) AS sub)
                 """, (session_id, session_id, self.max_rounds))
            
            self.mysql_client.connection.commit()
            
            self.logger.info(f"会话 {session_id} 历史更新成功")
            
            return history
        except pymysql.MySQLError as e:
            
            self.logger.error(f"更新会话历史失败: {e}")
            
            self.mysql_client.connection.rollback()
            
            raise
        except Exception as e:
            
            self.logger.error(f"更新会话历史意外错误: {e}")
            
            self.mysql_client.connection.rollback()
            
            raise

    
    def clear_session_history(self, session_id: str) -> bool:
        """
        函数功能: 清除指定会话历史
        :param session_id: 会话唯一标识(字符串)
        :return: True -> 清除成功, False -> 清除失败
        """
        try:
            
            self.mysql_client.cursor.execute("""
                 DELETE
                 FROM conversations
                 WHERE session_id = %s
                 """, (session_id,)
            )
            
            self.mysql_client.connection.commit()
            
            self.logger.info(f"会话 {session_id} 历史已清除")
            
            return True
        except pymysql.MySQLError as e:
            
            self.logger.error(f"清除会话历史失败: {e}")
            
            self.mysql_client.connection.rollback()
            
            return False
