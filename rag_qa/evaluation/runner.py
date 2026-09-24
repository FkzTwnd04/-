"""
执行离线 RAG 评测、保存逐题结果，并可与历史基线比较。

默认通过项目现有 RAGSystem 运行问题；也可注入兼容的测试对象，方便开发者单独验证评测编排。
本模块不会自动创建评测标注，也不会把模型生成结果当成正确答案。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from base.config import config
from rag_qa.evaluation.dataset import EvaluationSample, load_dataset
from rag_qa.evaluation.metrics import aggregate_intent_metrics, aggregate_metrics


QUALITY_METRICS = (
    "answer_exact_match",
    "answer_token_f1",
    "retrieval_hit_rate_at_k",
    "retrieval_recall_at_k",
    "retrieval_precision_at_k",
    "retrieval_mrr_at_k",
    "intent_accuracy",
    "intent_macro_f1",
    "intent_selective_accuracy",
)


def _safe_name(
    value: str,  # 调用者提供的运行名称；会清理成适合用作报告文件名的字符串。
) -> str:
    """把运行名限制成安全的文件名片段，避免路径穿越和非法字符。"""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not safe or safe in {".", ".."}:
        raise ValueError("run_name 不能为空或路径标记")
    return safe[:80]


def _chunk_to_record(
    chunk: Any,  # RAG 检索器返回的单个 Chunk；从中提取可写入 JSON 的字段。
) -> Dict[str, Any]:
    """抽取评测报告所需的 Chunk 信息，避免序列化框架对象本身。"""

    return {
        "chunk_id": getattr(chunk, "chunk_id", "") or getattr(chunk, "dedup_key", ""),
        "doc_id": getattr(chunk, "doc_id", ""),
        "source": getattr(chunk, "source", ""),
        "page": getattr(chunk, "page", None),
        "score": float(getattr(chunk, "score", 0.0) or 0.0),
        "text": getattr(chunk, "text", ""),
    }


def _build_default_system():
    """延迟加载重型模型和 Milvus 依赖，只有实际启动评测时才创建问答系统。"""

    from rag_qa.generation.llm import LLMClient
    from rag_qa.pipeline.qa import RAGSystem
    from rag_qa.retriever.hybrid import HybridRetriever

    return RAGSystem(HybridRetriever(), LLMClient())


class RAGEvaluationRunner:
    """将标注样本逐条交给 RAG 系统运行并生成可复查的评测报告。"""

    def __init__(
        self,
        rag_system=None,  # 可选的 RAG 问答系统实例；不传时在 RAG 评测启动时创建默认实例。
        intent_predictor=None,  # 可选的“问题 -> 意图标签或 None”函数，用于评测自定义意图方案。
        retrieval_k: Optional[int] = None,  # 检索指标统计的前 K 个结果数；缺省时使用项目配置。
    ):
        # 依赖注入使评测器可复用真实 RAG 系统，也可在开发时传入轻量替身。
        self.rag_system = rag_system
        self.intent_predictor = intent_predictor
        self.retrieval_k = retrieval_k or config.RETRIEVAL_K

    def _get_system(self):
        """首次执行评测时才初始化模型与向量库，避免仅查看 CLI 帮助就加载重资源。"""

        if self.rag_system is None:
            self.rag_system = _build_default_system()
        if not callable(getattr(self.rag_system, "answer_with_details", None)):
            raise TypeError("rag_system 必须提供 answer_with_details(query, source_filter=...) 方法")
        return self.rag_system

    def run(
        self,
        dataset_path: str,  # 待读取的 JSONL 评测集路径；其中的问题和人工标注决定可计算哪些指标。
        mode: str = "rag",  # 评测模式；rag 跑完整问答链路，intent 只评估意图预测结果。
        output_dir: str = "rag_qa/evaluation/results",  # 每次运行的 JSON 报告保存目录。
        run_name: str = "rag_eval",  # 本次运行的名称；用于报告文件名和基线标识。
        save_baseline: bool = False,  # 是否把本次报告另存为可供以后比较的基线文件。
        compare_baseline: Optional[str] = None,  # 可选的历史基线 JSON 路径；提供后会检查质量或耗时退化。
        max_quality_drop: float = 0.03,  # 允许质量指标下降的最大绝对值；0.03 表示最多下降 3 个百分点。
        max_latency_increase: float = 0.20,  # 允许平均耗时增加的最大比例；0.20 表示最多增加 20%。
    ) -> Dict[str, Any]:
        """运行数据集、保存报告，并按可选门槛检查是否相对基线退化。"""

        if max_quality_drop < 0 or max_latency_increase < 0:
            raise ValueError("回归阈值必须是非负数")

        if mode not in {"rag", "intent"}:
            raise ValueError("mode 只能是 'rag' 或 'intent'")

        samples = load_dataset(dataset_path)
        dataset_bytes = Path(dataset_path).read_bytes()
        dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
        safe_run_name = _safe_name(run_name)
        started_at = datetime.now(timezone.utc)

        # 两种评测复用同一份数据读取和报告结构，但分别调用 RAG 或意图预测能力。
        if mode == "rag":
            records = self._run_samples(self._get_system(), samples)
            metrics = aggregate_metrics(records, self.retrieval_k)
        else:
            records = self._run_intent_samples(samples)
            metrics = aggregate_intent_metrics(records)

        report = {
            "run_name": safe_run_name,
            "mode": mode,
            "created_at": started_at.isoformat(),
            "dataset_path": str(Path(dataset_path).resolve()),
            "dataset_sha256": dataset_hash,
            "configuration": {
                "llm_model": config.LLM_MODEL,
                "retrieval_k": self.retrieval_k,
                "candidate_m": config.CANDIDATE_M,
                "intent_predictor": (
                    "not_used"
                    if mode == "rag"
                    else "injected" if self.intent_predictor else "local_bert_classifier"
                ),
            },
            "metric_notes": {
                "answer_exact_match": "忽略大小写、空白和标点后的精确匹配，不等同于语义正确性。",
                "answer_token_f1": "英文按词、中文按字统计重合度，是字面指标，不是事实性评审。",
                "retrieval_metrics": "只统计提供了 relevant_chunk_ids 人工标注的样本。",
            },
            "metrics": metrics,
            "records": records,
        }

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        report_path = output_root / f"{safe_run_name}_{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
        _write_json(report_path, report)

        baseline_result = None
        if save_baseline:
            baseline_dir = output_root / "baselines"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            baseline_path = baseline_dir / f"{safe_run_name}.json"
            _write_json(baseline_path, report)
            baseline_result = {"saved": True, "path": str(baseline_path.resolve())}

        comparison = None
        if compare_baseline:
            baseline_data = _read_json(Path(compare_baseline))
            comparison = compare_reports(
                report,
                baseline_data,
                max_quality_drop=max_quality_drop,
                max_latency_increase=max_latency_increase,
            )
            report["baseline_comparison"] = comparison
            _write_json(report_path, report)

        return {
            "report_path": str(report_path.resolve()),
            "metrics": metrics,
            "baseline": baseline_result,
            "comparison": comparison,
        }

    def _run_samples(
        self,
        rag_system,  # 提供 answer_with_details 的 RAG 实例，用于获得答案和实际检索块。
        samples: List[EvaluationSample],  # 已校验的评测样本；每条样本会单独调用一次问答链路。
    ) -> List[Dict[str, Any]]:
        """逐条运行样本；单题异常记入报告，避免一题失败丢掉整批结果。"""

        records = []
        for index, sample in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] {sample.sample_id}: {sample.query}")
            started = time.perf_counter()
            record = {
                "id": sample.sample_id,
                "query": sample.query,
                "reference_answer": sample.reference_answer,
                "relevant_chunk_ids": list(sample.relevant_chunk_ids),
                "expected_intent": sample.expected_intent,
                "metadata": sample.metadata,
            }

            try:
                result = rag_system.answer_with_details(
                    sample.query,
                    source_filter=sample.source_filter,
                    history=None,
                )
                chunks = [_chunk_to_record(chunk) for chunk in result.get("chunks", [])]
                record.update(
                    {
                        "answer": result.get("answer", "") or "",
                        "category": result.get("category"),
                        "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                        "retrieved_chunks": chunks,
                        "latency_seconds": float(result.get("latency_seconds") or time.perf_counter() - started),
                    }
                )
            except Exception as error:
                # 保留异常类型和文本，方便定位坏样本；报告不包含完整 traceback 或密钥信息。
                record.update(
                    {
                        "answer": "",
                        "retrieved_chunk_ids": [],
                        "retrieved_chunks": [],
                        "latency_seconds": time.perf_counter() - started,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            records.append(record)

        return records

    def _run_intent_samples(
        self,
        samples: List[EvaluationSample],  # 必须带 expected_intent 的样本列表，用于逐题比较预测标签。
    ) -> List[Dict[str, Any]]:
        """在同一批人工标注问句上评估意图预测准确度、覆盖率和延迟。"""

        unlabeled = [sample.sample_id for sample in samples if not sample.expected_intent]
        if unlabeled:
            raise ValueError(
                "意图模式下每条样本都必须填写 expected_intent；缺少标注的样本："
                + ", ".join(unlabeled[:5])
            )

        predictor = self.intent_predictor
        if predictor is None:
            # 先为现有 BERT 分类器提供可运行基线；后续可注入 BM25+向量预测函数做公平比较。
            from rag_qa.router.classifier import QueryClassifier

            classifier = QueryClassifier()
            predictor = classifier.predict_category

        records = []
        for index, sample in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] {sample.sample_id}: {sample.query}")
            started = time.perf_counter()
            record = {
                "id": sample.sample_id,
                "query": sample.query,
                "expected_intent": sample.expected_intent,
                "metadata": sample.metadata,
            }
            try:
                # 预测器返回 None 表示该方案选择弃权，可用于评估覆盖率与低置信回退策略。
                prediction = predictor(sample.query)
                if prediction is not None and not isinstance(prediction, str):
                    raise TypeError("意图预测器必须返回字符串或 None")
                record["predicted_intent"] = prediction
                record["latency_seconds"] = time.perf_counter() - started
            except Exception as error:
                record.update(
                    {
                        "predicted_intent": None,
                        "latency_seconds": time.perf_counter() - started,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            records.append(record)
        return records


def compare_reports(
    current: Dict[str, Any],  # 本次评测产生的新报告；其中的指标作为当前版本结果。
    baseline: Dict[str, Any],  # 以前保存的基线报告；作为质量和耗时的比较参照。
    max_quality_drop: float = 0.03,  # 可接受的质量分数绝对下降范围，例如 0.03 即 3 个百分点。
    max_latency_increase: float = 0.20,  # 可接受的平均延迟相对增长范围，例如 0.20 即 20%。
) -> Dict[str, Any]:
    """比较质量指标和耗时；只有存在有效的双方指标时才执行相应门槛。"""

    current_metrics = current.get("metrics", {})
    baseline_metrics = baseline.get("metrics", {})
    regressions = []

    for metric_name in QUALITY_METRICS:
        current_value = current_metrics.get(metric_name)
        baseline_value = baseline_metrics.get(metric_name)
        if current_value is None or baseline_value is None:
            continue
        if current_value < baseline_value - max_quality_drop:
            regressions.append(
                {
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "allowed_drop": max_quality_drop,
                }
            )

    current_latency = current_metrics.get("mean_latency_seconds")
    baseline_latency = baseline_metrics.get("mean_latency_seconds")
    if current_latency is not None and baseline_latency and current_latency > baseline_latency * (1 + max_latency_increase):
        regressions.append(
            {
                "metric": "mean_latency_seconds",
                "baseline": baseline_latency,
                "current": current_latency,
                "allowed_relative_increase": max_latency_increase,
            }
        )

    return {
        "passed": not regressions,
        "baseline_run_name": baseline.get("run_name"),
        "max_quality_drop": max_quality_drop,
        "max_latency_increase": max_latency_increase,
        "regressions": regressions,
    }


def _write_json(
    path: Path,  # 要写入的报告或基线文件路径。
    value: Dict[str, Any],  # 要序列化为 JSON 的报告内容。
) -> None:
    """以 UTF-8 保存可读 JSON，便于人工检查每道题的输出和检索证据。"""

    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(
    path: Path,  # 已保存的基线报告路径；读取后检查是否含有指标对象。
) -> Dict[str, Any]:
    """载入基线报告并检查顶层格式，错误时指出基线文件路径。"""

    if not path.is_file():
        raise FileNotFoundError(f"找不到基线报告：{path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict) or not isinstance(value.get("metrics"), dict):
        raise ValueError(f"基线报告缺少 metrics 对象：{path}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数，提供评测运行、基线保存和回归比较入口。"""

    parser = argparse.ArgumentParser(description="运行 RAG 离线评测并保存逐题结果")
    parser.add_argument("--dataset", required=True, help="UTF-8 JSONL 评测集路径")
    parser.add_argument("--mode", choices=("rag", "intent"), default="rag", help="评测完整 RAG 或意图分类")
    parser.add_argument("--run-name", default="rag_eval", help="本次运行名称")
    parser.add_argument("--output-dir", default="rag_qa/evaluation/results", help="报告输出目录")
    parser.add_argument("--save-baseline", action="store_true", help="将本次报告另存为基线")
    parser.add_argument("--compare-baseline", help="与指定 JSON 基线报告比较")
    parser.add_argument("--max-quality-drop", type=float, default=0.03, help="允许的质量指标绝对下降值")
    parser.add_argument("--max-latency-increase", type=float, default=0.20, help="允许的平均耗时相对增幅")
    return parser


def main(
    argv: Optional[List[str]] = None,  # 可选命令行参数列表；None 时由 argparse 读取终端输入。
) -> int:
    """CLI 主入口；若回归比较失败则返回非零状态，便于接入后续自动化。"""

    args = _build_parser().parse_args(argv)
    runner = RAGEvaluationRunner()
    result = runner.run(
        dataset_path=args.dataset,
        mode=args.mode,
        output_dir=args.output_dir,
        run_name=args.run_name,
        save_baseline=args.save_baseline,
        compare_baseline=args.compare_baseline,
        max_quality_drop=args.max_quality_drop,
        max_latency_increase=args.max_latency_increase,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("comparison") and not result["comparison"]["passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
