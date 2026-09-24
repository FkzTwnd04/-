"""
维护意图识别使用的代表问句及其业务标签。

每条代表问句绑定一个 intent，可选绑定标准答案；是否允许直接返回该答案由
answer_reusable 控制。数据按 tenant_id 隔离，为意图检索器提供统一的数据入口。
"""

import argparse
import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from base.logger import logger


class IntentSampleStore:
    """通过 MySQL 保存意图代表问句，供检索器建立和刷新索引。"""

    def __init__(
        self,
        mysql_client,  # 已建立连接的 MySQL 客户端，负责执行 SQL 和提交事务。
        tenant_id: str = "default",  # 未接入鉴权前使用的单租户默认值；不能当作可信多租户身份。
    ):
        """
        函数功能: 绑定 MySQL 客户端和租户标识(为空时回退为 "default"), 并创建保护共享游标的可重入锁.
        :param mysql_client: 已连接的 MySQL 客户端
        :param tenant_id: 默认数据租户
        :return: None
        """
        self.mysql = mysql_client
        self.tenant_id = tenant_id or "default"
        self._lock = RLock()  # 同一进程内串行保护共享游标，避免并发 SQL 互相覆盖结果。

    def init_table(self) -> None:
        """创建意图问句表；重复调用不会重建或删除现有数据。"""

        sql = """
            CREATE TABLE IF NOT EXISTS intent_samples (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                sample_id VARCHAR(128) NOT NULL,
                query_text VARCHAR(2000) NOT NULL,
                intent VARCHAR(128) NOT NULL,
                standard_answer TEXT NULL,
                answer_reusable TINYINT(1) NOT NULL DEFAULT 0,
                is_active TINYINT(1) NOT NULL DEFAULT 1,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uq_intent_sample_tenant_id (tenant_id, sample_id),
                KEY idx_intent_sample_lookup (tenant_id, is_active, intent)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """
        with self._lock:
            self.mysql.cursor.execute(sql)
            self.mysql.connection.commit()

    def list_active(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """读取指定租户的有效代表问句，并转换为检索器使用的字典格式。"""

        tenant = tenant_id or self.tenant_id  # 调用方没传租户时使用实例绑定值。
        sql = """
            SELECT sample_id, query_text, intent, standard_answer, answer_reusable
            FROM intent_samples
            WHERE tenant_id = %s AND is_active = 1
            ORDER BY id ASC
        """
        with self._lock:
            self.mysql.cursor.execute(sql, (tenant,))
            rows = self.mysql.cursor.fetchall()

        return [
            {
                "sample_id": row[0],  # 数据库里的稳定样本编号，供命中结果定位原始记录。
                "query": row[1],  # 用于 BM25 和向量匹配的代表问句文本。
                "intent": row[2],  # 命中该代表问句时应返回的业务意图标签。
                "standard_answer": row[3],  # 可选的标准回复；为空时继续走后续问答链路。
                "answer_reusable": bool(row[4]),  # 只有标记为真且通过安全规则时才允许直接回复。
            }
            for row in rows
        ]

    def upsert(
        self,
        sample_id: str,  # 一条代表问句的稳定编号；同租户下用于新增或更新同一条记录。
        query: str,  # 用户历史或人工整理的代表问法，不能为空。
        intent: str,  # 该问法对应的目标意图标签，必须由业务标注确认。
        standard_answer: Optional[str] = None,  # 可信的固定答复；仅在答案适用且审核通过时填写。
        answer_reusable: bool = False,  # 是否允许命中后复用固定答复；动态/个性化问题必须为 False。
        tenant_id: Optional[str] = None,  # 数据归属租户；目前应用未接鉴权，应继续使用默认单租户。
    ) -> None:
        """新增或更新一条代表问句；同租户、同 sample_id 时覆盖该条内容。"""

        if not sample_id.strip() or not query.strip() or not intent.strip():
            raise ValueError("sample_id、query 和 intent 都必须是非空字符串")
        if answer_reusable and not (standard_answer and standard_answer.strip()):
            raise ValueError("允许复用标准答案时必须同时提供 standard_answer")

        tenant = tenant_id or self.tenant_id
        sql = """
            INSERT INTO intent_samples
                (tenant_id, sample_id, query_text, intent, standard_answer, answer_reusable, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, 1)
            ON DUPLICATE KEY UPDATE
                query_text = VALUES(query_text),
                intent = VALUES(intent),
                standard_answer = VALUES(standard_answer),
                answer_reusable = VALUES(answer_reusable),
                is_active = 1
        """
        values = (
            tenant,
            sample_id.strip(),
            query.strip(),
            intent.strip(),
            standard_answer.strip() if standard_answer else None,
            int(answer_reusable),
        )
        with self._lock:
            self.mysql.cursor.execute(sql, values)
            self.mysql.connection.commit()
        logger.info("意图代表问句已保存：tenant=%s, sample_id=%s", tenant, sample_id)

    def deactivate(self, sample_id: str, tenant_id: Optional[str] = None) -> bool:
        """软删除一条代表问句；保留数据库记录以便审计和恢复。"""

        tenant = tenant_id or self.tenant_id
        sql = "UPDATE intent_samples SET is_active = 0 WHERE tenant_id = %s AND sample_id = %s"
        with self._lock:
            self.mysql.cursor.execute(sql, (tenant, sample_id))
            changed = self.mysql.cursor.rowcount > 0
            self.mysql.connection.commit()
        return changed

    def import_jsonl(
        self,
        path: str,  # UTF-8 JSONL 代表问句文件；每行需包含 id、query 和 intent。
        tenant_id: Optional[str] = None,  # 导入记录所属租户；省略时使用本存储对象绑定的租户。
    ) -> int:
        """校验并逐条导入代表问句；遇到错误会指出文件行号并停止导入。"""

        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(f"找不到代表问句文件：{source_path}")

        imported_count = 0
        with source_path.open("r", encoding="utf-8-sig") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"第 {line_number} 行 JSON 格式错误：{error.msg}") from error
                if not isinstance(sample, dict):
                    raise ValueError(f"第 {line_number} 行必须是 JSON 对象")

                sample_id = sample.get("id")
                query = sample.get("query")
                intent = sample.get("intent")
                reusable = sample.get("answer_reusable", False)
                if not all(isinstance(value, str) and value.strip() for value in (sample_id, query, intent)):
                    raise ValueError(f"第 {line_number} 行必须填写非空 id、query 和 intent")
                if not isinstance(reusable, bool):
                    raise ValueError(f"第 {line_number} 行的 answer_reusable 必须是 true 或 false")
                answer = sample.get("standard_answer")
                if answer is not None and not isinstance(answer, str):
                    raise ValueError(f"第 {line_number} 行的 standard_answer 必须是字符串")

                self.upsert(
                    sample_id=sample_id,
                    query=query,
                    intent=intent,
                    standard_answer=answer,
                    answer_reusable=reusable,
                    tenant_id=tenant_id,
                )
                imported_count += 1
        return imported_count


def main() -> int:
    """命令行导入入口；用于把经过人工确认的代表问句文件写入 MySQL。"""

    parser = argparse.ArgumentParser(description="导入意图识别代表问句")
    parser.add_argument("--dataset", required=True, help="代表问句 JSONL 文件路径")
    parser.add_argument("--tenant-id", default="default", help="代表问句所属租户")
    args = parser.parse_args()

    from storage.db import MySQLClient

    mysql = MySQLClient()
    try:
        store = IntentSampleStore(mysql, tenant_id=args.tenant_id)
        store.init_table()
        imported = store.import_jsonl(args.dataset)
        print(f"已导入 {imported} 条代表问句到租户 {args.tenant_id}")
    finally:
        mysql.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
