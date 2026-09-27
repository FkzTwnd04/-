# RAG 离线评测

## 这个模块解决什么问题

问答系统的回答读起来流畅，不代表检索到了正确资料，也不代表答案和参考内容一致。修改切分方式、检索参数、重排器或提示词之后，需要用同一批人工标注的问题重新运行，记录变化前后的结果。

本模块把这件事整理成固定流程：读取 JSONL 评测集、调用项目 RAG 问答链路、保存每题答案和检索块、汇总指标，并可与之前保存的基线比较。它不会自动创建正确答案，也不会把模型自评当作客观分数。

## 文件分工

- `dataset.py` 读取 JSON Lines 格式样本，检查问题、参考答案、相关 Chunk ID、意图标签和来源筛选条件。
- `metrics.py` 计算答案精确匹配、字词重合 F1、检索 Recall@K/Precision@K/MRR@K/命中率，以及意图准确率、宏平均 F1、覆盖率和选择性准确率。
- `runner.py` 支持 RAG 问答和意图识别两种离线评测，保存逐题报告并比较历史基线。
- `baselines.md` 说明基线建立、比较原则和如何解读指标。
- `results/` 是运行时输出目录，保存每次评测报告；程序会自动创建。

## 准备评测数据

创建一个 UTF-8 编码的 `.jsonl` 文件，每一行是独立 JSON 对象。至少填写 `query`。要评测生成答案，需要填写人工核对的 `reference_answer`；要评测检索，需要填写 Milvus 中真实存在且人工确认相关的 `relevant_chunk_ids`。

```json
{"id":"faq-001","query":"这里填写一个真实问题","reference_answer":"这里填写人工核对的参考答案","relevant_chunk_ids":["Milvus 中相关文本块的真实 ID"],"source_filter":"ai"}
{"id":"faq-002","query":"另一个问题","reference_answer":"另一个参考答案","relevant_chunk_ids":["另一个真实文本块 ID"]}
```

`relevant_chunk_ids` 不能照抄示例占位文字，必须替换为你自己 Milvus 集合中的 Chunk ID。暂时没有标注相关块时可以省略该字段；这样该样本仍参与答案指标，但不会被当成检索失败。没有参考答案时，样本仍可参与检索指标，答案指标会跳过它。

建议先从实际知识库选 20 到 50 个常见问题，人工确认参考答案和支持答案的文本块。还要加入无相关资料、容易混淆、需要拒答以及不同来源过滤的问题。样本数量不是质量保证，标注正确和覆盖典型场景更重要。

## 运行一轮评测

在项目根目录、已安装项目依赖且 MySQL/Redis/Milvus/模型服务可用的 Python 环境中执行：

```powershell
python -m rag_qa.evaluation.runner --dataset path/to/your_eval.jsonl --run-name first-run
```

若项目使用了不同的 Python 环境，请将 `python` 替换为该环境的解释器路径。运行器会创建模型、检索器并调用现有 RAGSystem，因此需要可用的模型文件、Milvus 集合和大模型配置。每道题运行失败会被记入报告；其他题仍会继续评测。

报告保存在 `rag_qa/evaluation/results/`，内容包括运行配置、数据集 SHA-256、整体指标、逐题回答、检索块 ID、来源、页码、分数、耗时和错误。报告可能包含用户问题、检索文本和模型回答，应按项目数据敏感级别管理，不要未经检查上传或提交。

## 单独评估意图识别

意图评测要求每条数据提供人工确认的 `expected_intent`：

```json
{"id":"intent-001","query":"怎么重置我的密码","expected_intent":"账号问题"}
{"id":"intent-002","query":"介绍一下向量数据库","expected_intent":"知识问答"}
```

用项目现有 BERT 分类器建立基线：

```powershell
python -m rag_qa.evaluation.runner --mode intent --dataset rag_qa/evaluation/intent_eval.jsonl --run-name intent-bert-v1
```

意图检索器可作为 `query -> 标签或 None` 函数注入评测器，与 BERT 使用相同标注集比较。返回 `None` 表示弃权；报告同时展示覆盖率与接受预测后的准确率。BM25+向量模块已经接入在线问答，但尚无项目标注集结果，不能把 BERT 的分数当作检索器的评测结果。

## 保存和比较基线

确认评测数据、模型、索引和参数稳定后，保存基线：

```powershell
python -m rag_qa.evaluation.runner --dataset path/to/your_eval.jsonl --run-name rag-v1 --save-baseline
```

基线写入 `rag_qa/evaluation/results/baselines/rag-v1.json`。之后使用同一份数据集与新版本对比：

```powershell
python -m rag_qa.evaluation.runner --dataset path/to/your_eval.jsonl --run-name rag-v2 --compare-baseline rag_qa/evaluation/results/baselines/rag-v1.json
```

默认质量指标允许下降 0.03，平均延迟允许增加 20%。可用 `--max-quality-drop` 和 `--max-latency-increase` 调整门槛。基线比较失败时命令返回非零退出码，可供后续 CI 调用；当前仓库尚未配置自动 CI 门禁。

比较前要确认报告使用相同的数据集内容。程序会记录数据集哈希；模型、提示词、索引和参数也要检查是否一致。若改了模型或语料，应该把它作为有意的版本变化记录，而不是只看单个总分。

## 指标说明与边界

- **Answer Exact Match**：忽略大小写、空白和标点后，回答是否与参考答案完全相同。适合短事实，不适合表达方式多样的长答案。
- **Answer Token F1**：英文按词、中文按字计算答案和参考答案的重合程度。它是简单字面指标，不能判断同义表达、逻辑正确性或事实是否有依据。
- **Recall@K**：人工标注的相关 Chunk 中，有多少被前 K 个结果找回。
- **Precision@K**：前 K 个结果中相关 Chunk 的占比。
- **MRR@K**：第一个相关 Chunk 排名越靠前，分数越高。
- **Hit Rate@K**：前 K 个结果中是否至少包含一个人工标注的相关 Chunk。
- **Mean Latency**：端到端单题耗时均值，包含分类、检索、重排和生成等阶段；目前没有分阶段计时。

本实现不计算 Faithfulness、语义相关性或事实正确性，因为这些指标需要额外标注或独立评审器。不要把字面 F1 当作事实忠实度，也不要把评测分数写成线上服务质量保证。后续可在评测数据和基础指标稳定后接入 RAGAS 或经过人工校验的模型评审，并单独报告评审器设置。