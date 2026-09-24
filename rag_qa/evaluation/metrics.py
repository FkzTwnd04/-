"""
计算不依赖额外模型调用的基础检索与答案指标。

检索指标使用人工标注的相关 Chunk ID；答案指标比较模型回答与参考答案的字面重合。
字符级 F1 和精确匹配不是语义评审，也不能单独证明回答忠实或事实正确。
"""

from collections import Counter
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional


ANSWER_METRICS = (
    "answer_exact_match",
    "answer_token_f1",
)
RETRIEVAL_METRICS = (
    "retrieval_hit_rate_at_k",
    "retrieval_recall_at_k",
    "retrieval_precision_at_k",
    "retrieval_mrr_at_k",
)
INTENT_METRICS = (
    "intent_accuracy",
    "intent_macro_f1",
    "intent_coverage",
    "intent_selective_accuracy",
)


def normalize_answer(
    text: str,  # 待比较的答案文本；函数会统一大小写、空格和标点后再返回。
) -> str:
    """统一大小写和空白，并去掉标点符号，供保守的精确匹配使用。"""

    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace() and unicodedata.category(character)[0] not in {"P", "S"}
    )


def _answer_tokens(
    text: str,  # 要拆分成计分单元的文本；中文按字、英文按词拆分。
) -> List[str]:
    """英文按词、中文按字切分，减少额外分词依赖带来的环境差异。"""

    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", normalized)


def token_f1(
    prediction: str,  # 系统生成的答案；作为计算精确率的预测文本。
    reference: str,  # 人工核对的标准答案；作为计算召回率的参照文本。
) -> float:
    """计算词项多重集合的 F1；完全没有重合时返回 0。"""

    predicted_tokens = Counter(_answer_tokens(prediction))
    reference_tokens = Counter(_answer_tokens(reference))
    overlap = sum((predicted_tokens & reference_tokens).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(predicted_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _mean(
    values: Iterable[float],  # 一组可参与平均的指标分数；可以为空。
) -> Optional[float]:
    """空样本集合返回 None，避免把“没有可算数据”误报成 0 分。"""

    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def score_record(
    record: Dict[str, Any],  # 一道题的标准答案、相关块标注及系统输出组成的记录。
    k: int,  # 检索评估的截断位置；只统计排名前 K 的文本块。
) -> Dict[str, Optional[float]]:
    """为单条问答结果计算可用指标；缺少人工标注时对应值设为 None。"""

    scores: Dict[str, Optional[float]] = {}
    reference = record.get("reference_answer")
    prediction = record.get("answer") or ""
    if isinstance(reference, str) and reference.strip():
        scores["answer_exact_match"] = float(
            normalize_answer(prediction) == normalize_answer(reference)
        )
        scores["answer_token_f1"] = token_f1(prediction, reference)
    else:
        scores["answer_exact_match"] = None
        scores["answer_token_f1"] = None

    # 没有相关 Chunk 标注的样本不能用于判断检索好坏，不能按“未命中”计分。
    relevant_ids = set(record.get("relevant_chunk_ids") or [])
    if not relevant_ids:
        for metric_name in RETRIEVAL_METRICS:
            scores[metric_name] = None
        return scores

    retrieved_ids = list(record.get("retrieved_chunk_ids") or [])[:k]
    relevant_ranks = [rank for rank, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in relevant_ids]
    hit_count = len(set(retrieved_ids) & relevant_ids)
    scores["retrieval_hit_rate_at_k"] = float(hit_count > 0)
    scores["retrieval_recall_at_k"] = hit_count / len(relevant_ids)
    scores["retrieval_precision_at_k"] = hit_count / k if k else 0.0
    scores["retrieval_mrr_at_k"] = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    return scores


def aggregate_metrics(
    records: List[Dict[str, Any]],  # 本次 RAG 评测产生的逐题结果列表。
    k: int,  # 检索指标使用的排名上限，与系统配置的 Recall@K 等 K 值一致。
) -> Dict[str, Any]:
    """汇总逐样本分数、失败数量和实际可参与计算的样本数。"""

    scored_records = [score_record(record, k) for record in records]
    metric_names = (*ANSWER_METRICS, *RETRIEVAL_METRICS)
    metrics = {
        name: _mean(score[name] for score in scored_records if score[name] is not None)
        for name in metric_names
    }

    return {
        "sample_count": len(records),
        "failed_sample_count": sum(bool(record.get("error")) for record in records),
        "answer_labeled_count": sum(bool(record.get("reference_answer")) for record in records),
        "retrieval_labeled_count": sum(bool(record.get("relevant_chunk_ids")) for record in records),
        "mean_latency_seconds": _mean(
            record["latency_seconds"] for record in records if record.get("latency_seconds") is not None
        ),
        **metrics,
    }


def aggregate_intent_metrics(
    records: List[Dict[str, Any]],  # 每题包含人工 expected_intent 和预测 predicted_intent 的结果列表。
) -> Dict[str, Any]:
    """计算意图预测质量，并单独报告模型拒绝判断时的覆盖率。"""

    labeled = [record for record in records if record.get("expected_intent")]
    if not labeled:
        raise ValueError("意图评测至少需要一条包含 expected_intent 的样本")

    labels = sorted({record["expected_intent"] for record in labeled})
    correct = sum(record.get("predicted_intent") == record["expected_intent"] for record in labeled)
    accepted = [record for record in labeled if record.get("predicted_intent") is not None]
    selective_correct = sum(
        record["predicted_intent"] == record["expected_intent"] for record in accepted
    )

    per_label_f1 = []
    predicted_labels = sorted(
        {record["predicted_intent"] for record in labeled if record.get("predicted_intent")}
    )
    matrix_labels = sorted(set(labels) | set(predicted_labels) | {"__abstain__"})
    confusion: Dict[str, Dict[str, int]] = {
        label: {candidate: 0 for candidate in matrix_labels} for label in labels
    }
    for record in labeled:
        expected = record["expected_intent"]
        predicted = record.get("predicted_intent") or "__abstain__"
        confusion[expected][predicted] += 1

    # 对每个真实类别分别计算 F1 后做宏平均，避免高频意图掩盖少数类的表现。
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in labels if other != label
        )
        false_negative = sum(
            confusion[label][other] for other in matrix_labels if other != label
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_label_f1.append(2 * true_positive / denominator if denominator else 0.0)

    total_count = len(labeled)
    return {
        "sample_count": len(records),
        "labeled_count": total_count,
        "failed_sample_count": sum(bool(record.get("error")) for record in records),
        "intent_accuracy": correct / total_count,
        "intent_macro_f1": sum(per_label_f1) / len(per_label_f1),
        "intent_coverage": len(accepted) / total_count,
        "intent_selective_accuracy": (
            selective_correct / len(accepted) if accepted else None
        ),
        "mean_latency_seconds": _mean(
            record["latency_seconds"] for record in labeled if record.get("latency_seconds") is not None
        ),
        "confusion_matrix": confusion,
        "labels": labels,
    }
