"""
读取并校验离线评测样本。

每条样本至少要给出问题；可以提供参考答案、相关 Chunk ID、来源过滤值和自定义信息。
评测程序只对人工填写的参考答案或相关 Chunk 计算指标，不会把模型自己的输出当标准答案。
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EvaluationSample:
    """一条不可变的评测记录，避免运行期间意外改写标注数据。"""

    sample_id: str  # 当前样本的唯一编号；用于报告定位样本并检查重复数据。
    query: str  # 要交给 RAG 系统或意图预测器处理的用户问题。
    reference_answer: Optional[str] = None  # 人工核对的标准答案；缺省时跳过答案质量指标。
    relevant_chunk_ids: Tuple[str, ...] = ()  # 人工确认支持答案的文本块 ID；缺省时跳过检索质量指标。
    source_filter: Optional[str] = None  # 可选的知识来源过滤条件；传给 RAG 检索器限制搜索范围。
    expected_intent: Optional[str] = None  # 人工确认的正确意图标签；意图评测用它对照预测结果。
    metadata: Dict[str, Any] = field(default_factory=dict)  # 除固定字段外保留的自定义标注，随逐题结果写入报告。


def _parse_sample(
    raw: Any,  # JSONL 当前行解析出的对象，后续从中读取问题、标签和其他标注字段。
    line_number: int,  # 当前记录在文件中的行号；用于报错时指出需要修正的位置。
) -> EvaluationSample:
    """把 JSON 对象转成强类型样本，并在报错中指出 JSONL 行号。"""

    if not isinstance(raw, dict):
        raise ValueError(f"第 {line_number} 行必须是 JSON 对象")

    query = raw.get("query")  # 取出待评测的问题文本；它是每条样本唯一必需的业务内容。
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"第 {line_number} 行的 query 必须是非空字符串")

    sample_id = raw.get("id") or f"sample-{line_number:04d}"  # 优先使用数据自带编号；没写时用行号生成稳定的默认编号。
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"第 {line_number} 行的 id 必须是非空字符串")

    reference_answer = raw.get("reference_answer")  # 读取人工标准答案，后续用于计算生成答案的字面指标。
    if reference_answer is not None and not isinstance(reference_answer, str):
        raise ValueError(f"第 {line_number} 行的 reference_answer 必须是字符串")

    relevant_ids = raw.get("relevant_chunk_ids", [])  # 读取人工标注的相关块 ID，后续用于评估检索结果。
    if not isinstance(relevant_ids, list) or any(not isinstance(item, str) for item in relevant_ids):
        raise ValueError(f"第 {line_number} 行的 relevant_chunk_ids 必须是字符串数组")

    source_filter = raw.get("source_filter")  # 读取检索来源限制，例如只允许从指定知识库来源召回。
    if source_filter is not None and not isinstance(source_filter, str):
        raise ValueError(f"第 {line_number} 行的 source_filter 必须是字符串")

    expected_intent = raw.get("expected_intent")  # 读取人工正确意图；意图评测模式要求每条样本都提供此值。
    if expected_intent is not None and (not isinstance(expected_intent, str) or not expected_intent.strip()):
        raise ValueError(f"第 {line_number} 行的 expected_intent 必须是非空字符串")

    # 未被本模块消费的键保留为 metadata，方便保存意图标签、场景名等扩展标注。
    known_fields = {  # 列出本模块有专门字段接收的 JSON 键；其他键会归入 metadata。
        "id",  # 样本编号由 sample_id 接收，不重复存入扩展信息。
        "query",  # 问题文本由 query 接收，不重复存入扩展信息。
        "reference_answer",  # 标准答案由 reference_answer 接收，不重复存入扩展信息。
        "relevant_chunk_ids",  # 相关文本块编号由 relevant_chunk_ids 接收，不重复存入扩展信息。
        "source_filter",  # 来源过滤条件由 source_filter 接收，不重复存入扩展信息。
        "expected_intent",  # 标准意图标签由 expected_intent 接收，不重复存入扩展信息。
    }
    metadata = {key: value for key, value in raw.items() if key not in known_fields}  # 保留调用方自定义字段，避免扩展标注在加载时丢失。

    return EvaluationSample(
        sample_id=sample_id.strip(),  # 去掉编号两端空白，保证同一编号比较时不受格式空格影响。
        query=query.strip(),  # 去掉问题两端空白，避免空格影响检索和报告展示。
        reference_answer=reference_answer,  # 将人工标准答案放入样本，供问答评测计算答案指标。
        relevant_chunk_ids=tuple(dict.fromkeys(relevant_ids)),  # 转为不可变元组并去重，同时保留人工标注顺序。
        source_filter=source_filter,  # 保留来源限制，运行 RAG 评测时传给检索环节。
        expected_intent=expected_intent.strip() if expected_intent else None,  # 清理标签空格；未填写时保留 None 供 RAG 模式使用。
        metadata=metadata,  # 把额外标注一并带入报告，便于按场景或其他自定义维度分析。
    )


def load_dataset(
    path: str,  # 评测集 JSONL 文件路径；相对路径以启动命令时的项目目录为基准。
) -> List[EvaluationSample]:
    """逐行读取 UTF-8 JSONL 数据集，并拒绝空文件、坏 JSON 和重复样本 ID。"""

    dataset_path = Path(path)  # 把字符串路径转换为 Path，便于检查文件并按行读取。
    if not dataset_path.is_file():
        raise FileNotFoundError(f"找不到评测集文件：{dataset_path}")

    samples: List[EvaluationSample] = []  # 按文件顺序保存校验通过的样本，作为评测器的输入列表。
    seen_ids = set()  # 记录已经出现的样本编号，用于拒绝重复 ID。
    with dataset_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"第 {line_number} 行 JSON 格式错误：{error.msg}") from error

            sample = _parse_sample(raw, line_number)
            if sample.sample_id in seen_ids:
                raise ValueError(f"第 {line_number} 行样本 id 重复：{sample.sample_id}")
            seen_ids.add(sample.sample_id)
            samples.append(sample)

    if not samples:
        raise ValueError(f"评测集为空：{dataset_path}")
    return samples
