"""装配 MySQL 热点问答、Jev 分类、RAG、会话和运营存储，并协调在线请求。

问候返回通用固定话术；其他问题先经 BM25 检查 jpkb，可靠命中直接使用数据库答案，
未命中再由 Jev 选择专业 RAG 或通用大模型。模型回答及实际引用文件写入会话和运营记录。
"""

import time
import uuid
import warnings
import os

warnings.filterwarnings("ignore")

from storage import MySQLClient

from storage.conversations import ConversationStore
from storage.ops import OpsStore
from storage.subject_qa import SubjectQASearch

from rag_qa.pipeline.qa import RAGSystem

from rag_qa.retriever.hybrid import HybridRetriever

from base.config import config
from base.logger import logger

from rag_qa.generation.llm import LLMClient
from rag_qa.router.intent import check_greeting
from rag_qa.router.jev_classifier import JevIntentClassifier

class IntegratedQASystem:
    
    def __init__(self):
        """
        函数功能: 装配 MySQL 客户端、热点问答检索、运营存储、大模型客户端、混合检索器、RAG 系统(含 Jev 分类器)和会话存储, 并初始化会话表; 大模型客户端初始化失败时记录错误后重新抛出异常.
        :return: None
        """
        
        self.logger = logger
        self.config = config
        
        self.mysql_client = MySQLClient()
        self.subject_qa_search = SubjectQASearch(self.mysql_client)
        self.ops_store = OpsStore(self.mysql_client)
        
        try:
            self.llm = LLMClient()
        except Exception as e:
            
            self.logger.error(f"大模型客户端初始化失败: {e}")
            
            raise

        
        self.vector_store = HybridRetriever()  
        self.rag_system = RAGSystem(
            self.vector_store,
            self.call_dashscope,
            classifier=JevIntentClassifier(),
        )

        
        
        
        self.conversations = ConversationStore(self.mysql_client, self.logger)
        self.init_conversation_table()

    
    def init_conversation_table(self):
        """
        函数功能: 初始化MySQL中的 Conversations表, 用于存储对话历史.
        :return:
        """
        
        return self.conversations.init_conversation_table()

    
    def call_dashscope(self, prompt):
        """
        函数功能: 调用大模型生成答案 -> 流式输出
        :param prompt: 发送给大模型的提示文本(字符串), 包含用户问题及上下文
        :return: 生成器, 逐段返回大模型生成的答案内容(字符串)
        """
        
        
        
        yield from self.llm.stream(prompt)

    
    def _fetch_recent_history(self, session_id):
        """
        函数功能: 获取最近5轮对话历史
        :param session_id: 会话唯一标识(字符串), 用于定位具体会话.
        :return: 列表, 包含字典形式的对话记录, 例如: [{'question':问题, 'answer':答案}]
        """
        
        return self.conversations._fetch_recent_history(session_id)

    
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
        
        return self.conversations.update_session_history(session_id, question, answer)

    
    def clear_session_history(self, session_id: str) -> bool:
        """
        函数功能: 清除指定会话历史
        :param session_id: 会话唯一标识(字符串)
        :return: True -> 清除成功, False -> 清除失败
        """
        
        return self.conversations.clear_session_history(session_id)

    
    def lookup_reusable_answer(
        self,
        query,  # 当前用户输入；先与 jpkb 热点问答做 BM25 匹配，再记录代表问句匹配情况。
        tenant_id="default",  # 可信调用方提供的租户范围；API 尚未接入身份鉴权时保持默认单租户。
    ):
        """将问题与 jpkb 热点问答做 BM25 比较；达到阈值时返回答案，未命中交给 Jev。"""

        hot_answer = self.subject_qa_search.search(query)
        if hot_answer:
            return hot_answer["answer"], hot_answer["hit_layer"], None
        return None, None, None

    def _resolve_citation_filenames(self, citations, tenant_id="default"):
        """用文档台账将旧向量块的 doc_id 补成用户能识别的实际文件名。"""

        document_ids = sorted({item.get("document_id") for item in citations if item.get("document_id")})
        if not document_ids:
            return citations
        placeholders = ",".join(["%s"] * len(document_ids))
        try:
            self.mysql_client.cursor.execute(
                "SELECT doc_id, source_path, title FROM documents "
                f"WHERE tenant_id = %s AND doc_id IN ({placeholders})",
                [tenant_id or "default", *document_ids],
            )
            rows = self.mysql_client.cursor.fetchall()
        except Exception as error:
            self.mysql_client.connection.rollback()
            self.logger.info("无法从文档台账补充知识文件名，将显示向量块已有名称：%s", error)
            return citations
        names = {
            row[0]: (os.path.basename(row[1] or "") or row[2] or "")
            for row in rows
        }
        return [
            {**item, "filename": item.get("filename") or names.get(item.get("document_id"), "")}
            for item in citations
        ]

    def query(
        self,
        query,  # 要回答的用户问题。
        source_filter=None,  # 可选知识来源限制，会透传到 RAG 检索阶段。
        session_id=None,  # 可选会话编号；提供时会读取并保存该会话历史。
        tenant_id="default",  # 可信租户标识；未完成鉴权前只能使用默认单租户。
        request_id_callback=None,  # 可选回调；运营日志写入后用它把反馈编号交给 WebSocket/API。
        metadata_callback=None,  # 可选回调；把本轮路由类型和实际 RAG 引用文件传给 API 展示。
    ):
        """
        函数功能: 查询集成系统，支持对话历史和流式输出
        :param query:
        :param source_filter:
        :param session_id:
        :return:
        """
        
        start_time = time.time()
        
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")

        
        history = self.get_session_history(session_id) if session_id else []

        # 简单问候直接使用通用固定话术，避免被题库或 FAQ 的近似问句误命中。
        greeting_response = check_greeting(query)
        if greeting_response:
            answer, hit_layer, intent_match = greeting_response, "greeting", None
        else:
            answer, hit_layer, intent_match = self.lookup_reusable_answer(query, tenant_id=tenant_id)

        if answer:
            
            self.logger.info("复用答案层命中：%s", hit_layer)
            if metadata_callback:
                mode = "database" if hit_layer and hit_layer.startswith("jpkb_") else "greeting"
                metadata_callback({"answer_mode": mode, "citations": []})
            if session_id:
                
                self.update_session_history(session_id, query, answer)
            
            processing_time = time.time() - start_time
            request_id = self.ops_store.log_request(
                query=query, answer=answer, hit_layer=hit_layer or "L3",
                intent=intent_match.get("intent") if intent_match else None,
                latency_ms=int(processing_time * 1000),
            )
            if request_id_callback:
                request_id_callback(request_id)
            
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            
            yield answer, True
        else:
            
            self.logger.info("数据库热点问答未超过阈值，交由 Jev 判断走完整 RAG 或通用大模型")
            
            collected_answer = ""
            
            
            
            def collect_answer_details(details):
                """
                函数功能: 用文档台账把 RAG 返回的引用补全文件名并写回 details, 若提供 metadata_callback 则上报回答模式(general_llm 或 rag)和引用列表.
                :param details: RAG 系统回传的回答详情字典, 含 category 和 citations 等字段, 会被原地修改
                :return: None
                """
                details["citations"] = self._resolve_citation_filenames(
                    details.get("citations", []), tenant_id=tenant_id
                )
                if metadata_callback:
                    metadata_callback({
                        "answer_mode": "general_llm" if details.get("category") == "通用知识" else "rag",
                        "citations": details["citations"],
                    })

            for token in self.rag_system.generate_answer(
                query,
                source_filter=source_filter,
                history=history,
                stream=True,
                details_callback=collect_answer_details,
            ):
                
                collected_answer += token
                
                yield token, False
            if session_id:
                
                self.update_session_history(session_id, query, collected_answer)
            
            processing_time = time.time() - start_time
            request_id = self.ops_store.log_request(
                query=query, answer=collected_answer, hit_layer="L3",
                intent=intent_match.get("intent") if intent_match else None,
                latency_ms=int(processing_time * 1000),
            )
            if request_id_callback:
                request_id_callback(request_id)
            
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            
            yield "", True

def main():
    """
    函数功能: 命令行交互入口, 循环读取问题和可选学科类别, 流式打印答案及最近对话历史, 输入 exit 退出; 出现异常时记录并打印错误, 最后关闭 MySQL 连接.
    :return: None
    """
    
    qa_system = IntegratedQASystem()  
    
    session_id = str(uuid.uuid4())
    
    print("\n欢迎使用集成问答系统！")
    
    print(f"会话ID: {session_id}")
    
    print(f"支持的学科类别：{qa_system.config.VALID_SOURCES}")
    
    print("输入查询进行问答，输入 'exit' 退出。")

    try:
        while True:
            
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                
                logger.info("退出系统")
                
                print("再见, 感谢您的使用, 期待下次再会！")
                
                break
            
            
            source_filter = input(f"请输入学科类别 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                
                logger.warning(f"无效的学科类别 '{source_filter}'，将不过滤")
                
                source_filter = None
            
            print("\n答案: ", end="", flush=True)
            
            answer = ""
            
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:
                    
                    print(token, end="", flush=True)
                    
                    answer += token
                if is_complete:
                    
                    print()
                    break
            
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")
    except Exception as e:
        
        logger.error(f"系统错误: {e}")
        
        print(f"发生错误: {e}")
    finally:
        
        qa_system.mysql_client.close()

if __name__ == '__main__':
    main()
