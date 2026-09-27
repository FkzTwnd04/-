# 智学答 · 学习资料 RAG 问答系统

面向Agent专业知识的检索增强问答（RAG）系统。用户提问后，系统依次尝试：问候规则 → MySQL 热点问答库（BM25）→ Jev 问题分类 → 通用大模型或完整 RAG（查询改写、Milvus 稠密 + 稀疏混合检索、父块提升、BGE 重排、带引用的生成）。附带文档上传入库、离线评测和运营日志的 Web 管理页面。

模块设计和实现状态见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 目录

- [系统环境](#系统环境)
- [快速预览（不接外部服务）](#快速预览不接外部服务)
- [完整部署步骤](#完整部署步骤)
- [配置说明](#配置说明)
- [导入知识库文档](#导入知识库文档)
- [验证与测试](#验证与测试)
- [常见问题](#常见问题)

---

## 系统环境

### 软件

| 组件 | 版本 | 用途 |
|---|---|---|
| Python | **3.11**（推荐） | 后端。`requirements.txt` 锁定的 `Pillow==9.5.0` 等包没有 3.12+ 的预编译包 |
| Node.js | 20 LTS 及以上 | 仅构建前端时需要（Vite 6） |
| MySQL | 8.0 | 热点问答库 `jpkb`、会话历史、文档台账、运营日志 |
| Redis | 6 / 7 | 答案缓存 |
| Milvus | 2.5.x Standalone | 向量库（稠密 + 稀疏混合检索） |
| Docker Desktop | 可选 | 推荐用它启动 MySQL、Redis、Milvus |

已在 Windows 11 + PowerShell 下验证；Linux / macOS 同样适用，只需把命令换成对应 shell 写法。

### 外部 API

| 服务 | 是否必需 | 说明 |
|---|---|---|
| 阿里云百炼 DashScope | 必需 | 大模型生成（默认 `qwen-plus`，OpenAI 兼容接口） |
| OpenRouter（Jev） | 可选 | 专业问题 / 通用问题路由；没有密钥时专业问题一律走完整 RAG |

### 本地模型

模型放在 `rag_qa/models/` 下（已被 `.gitignore` 排除，需要自行下载）：

| 目录名 | 来源 | 是否必需 | 用途 |
|---|---|---|---|
| `bge-m3` | `BAAI/bge-m3` | 必需 | 稠密 + 稀疏向量编码 |
| `bge-reranker-large` | `BAAI/bge-reranker-large` | 必需 | 召回结果重排 |
| `nlp_bert_document-segmentation_chinese-base` | ModelScope `iic/nlp_bert_document-segmentation_chinese-base` | 可选 | 模型语义切分（默认不使用） |
| `bert-base-chinese`、`bert_query_classifier` | `google-bert/bert-base-chinese` + 自行训练 | 可选 | 旧版 BERT 分类器，仅离线对比 |

### 硬件建议

- 内存 ≥ 16 GB（bge-m3 与 reranker 同时加载约占 5 GB，另需 Milvus 和 MySQL）。
- 磁盘 ≥ 15 GB 可用空间（模型约 6.5 GB，Docker 镜像及数据另计）。
- 无 GPU 也能运行，CPU 上首次加载模型和单次检索会慢一些；装有 CUDA 版 PyTorch 时自动使用 GPU。

---

## 快速预览（不接外部服务）

只想看界面时，不需要数据库、模型和密钥：

```powershell
cd frontend
npm install
npm run dev
```

浏览器打开 <http://127.0.0.1:5173/static/>。页面可正常切换，但问答、上传等操作会因后端未启动而提示失败。

---

## 完整部署步骤

以下命令均在**项目根目录**的 PowerShell 中执行。

### 1. 创建 Python 环境并安装依赖

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 如果 PowerShell 提示“禁止运行脚本”，先执行一次 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。
> 需要 GPU 时，按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 的命令改装 CUDA 版 `torch`。

### 2. 下载模型

任选一种方式（国内网络推荐 ModelScope）：

```powershell
# 方式 A：ModelScope
modelscope download --model BAAI/bge-m3 --local_dir rag_qa/models/bge-m3
modelscope download --model BAAI/bge-reranker-large --local_dir rag_qa/models/bge-reranker-large

# 方式 B：Hugging Face（国内可先设置镜像 $env:HF_ENDPOINT="https://hf-mirror.com"）
huggingface-cli download BAAI/bge-m3 --local-dir rag_qa/models/bge-m3
huggingface-cli download BAAI/bge-reranker-large --local-dir rag_qa/models/bge-reranker-large
```

完成后应有 `rag_qa/models/bge-m3/config.json` 和 `rag_qa/models/bge-reranker-large/config.json`。

### 3. 启动 MySQL、Redis、Milvus

使用 Docker（把 `你的MySQL密码`、`你的Redis密码` 换成自己的，之后填进 `.env`）：

```powershell
# MySQL 8.0，自动创建 subjects_kg 库
docker run -d --name zhixueda-mysql -p 3306:3306 `
  -e MYSQL_ROOT_PASSWORD=你的MySQL密码 -e MYSQL_DATABASE=subjects_kg `
  mysql:8.0 --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci

# Redis 7（不想设密码就去掉 --requirepass 部分，并让 REDIS_PASSWORD 留空）
docker run -d --name zhixueda-redis -p 6379:6379 redis:7 redis-server --requirepass 你的Redis密码

# Milvus 2.5 Standalone
Invoke-WebRequest https://github.com/milvus-io/milvus/releases/download/v2.5.4/milvus-standalone-docker-compose.yml -OutFile milvus-compose.yml
docker compose -f milvus-compose.yml up -d
```

已有本机安装的 MySQL / Redis / Milvus 也可以直接用，只要在 `.env` 里填对地址和密码。

### 4. 填写 `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

至少填写以下几项，其余保持默认即可（完整说明见[配置说明](#配置说明)）：

```ini
DASHSCOPE_API_KEY=sk-xxxxxxxx        # 百炼控制台创建的 API Key
MYSQL_PASSWORD=你的MySQL密码
REDIS_PASSWORD=你的Redis密码          # 无密码留空
JEV_API_KEY=sk-or-xxxxxxxx           # 可选，OpenRouter 密钥
```

检查配置是否读取成功（只显示“是否已配置”，不打印密钥）：

```powershell
python -m base.config
```

输出中 `缺失的必需配置: []` 表示必需的密钥都已就绪；可选的 `JEV_API_KEY` 未填写时显示为“未配置(可选)”。

### 5. 初始化数据

**Milvus 数据库**：集合会在首次使用时自动创建，但数据库需要先建好：

```powershell
python -c "from pymilvus import MilvusClient; c = MilvusClient(uri='http://localhost:19530'); 'itcast' in c.list_databases() or c.create_database('itcast'); print(c.list_databases())"
```

**MySQL 表**：`conversations`、`documents`、`request_logs`、`faq_entries` 等表由程序首次使用时自动创建。热点问答表 `jpkb` 需要手动建表并导入题库，CSV 需包含 `学科名称`、`问题`、`答案` 三列：

```powershell
python -c "from storage.db import MySQLClient; c = MySQLClient(); c.create_table(); c.insert_data(r'路径\jpkb.csv'); c.close()"
```

没有题库时只建表（去掉 `insert_data` 部分）即可，所有问题会直接进入 Jev / RAG 流程。

### 6. 构建前端

```powershell
cd frontend
npm install
npm run build
cd ..
```

构建结果写入 `static/`，由后端直接提供。仓库中已附带一份构建好的 `static/`，未修改前端时可跳过这一步。

### 7. 启动服务

```powershell
python -m uvicorn api.server:app --host 127.0.0.1 --port 8080
```

| 地址 | 内容 |
|---|---|
| <http://127.0.0.1:8080> | Web 页面（问答 / 知识库 / 评测 / 运营） |
| <http://127.0.0.1:8080/docs> | 接口文档（Swagger） |
| <http://127.0.0.1:8080/api/system/status> | 服务状态与密钥配置情况 |

服务启动时不会连接数据库和加载模型；第一次提问时才初始化，因此**首个问题需要等待几十秒**，之后恢复正常速度。日志输出到终端和 `logs/app.log`。

开发前端时可以同时开两个终端：一个运行上面的后端，另一个在 `frontend/` 执行 `npm run dev`，访问 <http://127.0.0.1:5173/static/>，接口请求会被代理到 8080 端口。

---

## 配置说明

所有配置都在 `.env` 中，优先级为：**系统环境变量 > `.env` > 代码默认值**（`base/config.py`）。`.env` 已被 `.gitignore` 排除，不会提交；可提交的模板是 `.env.example`。

| 分组 | 主要变量 | 默认值 |
|---|---|---|
| 密钥 | `DASHSCOPE_API_KEY`、`JEV_API_KEY`、`MYSQL_PASSWORD`、`REDIS_PASSWORD` | 无 |
| MySQL | `MYSQL_HOST`、`MYSQL_USER`、`MYSQL_DATABASE` | `localhost`、`root`、`subjects_kg` |
| Redis | `REDIS_HOST`、`REDIS_PORT`、`REDIS_DB` | `localhost`、`6379`、`0` |
| Milvus | `MILVUS_HOST`、`MILVUS_PORT`、`MILVUS_DATABASE_NAME`、`MILVUS_COLLECTION_NAME` | `localhost`、`19530`、`itcast`、`edurag_final` |
| 大模型 | `LLM_MODEL`、`DASHSCOPE_BASE_URL` | `qwen-plus`、百炼兼容地址 |
| Jev 路由 | `JEV_ENDPOINT`、`JEV_MODEL`、`JEV_CONFIDENCE_THRESHOLD` | OpenRouter、`typesafe/jev-1.13`、`0.7` |
| 检索 | `PARENT_CHUNK_SIZE`、`CHILD_CHUNK_SIZE`、`CHUNK_OVERLAP`、`RETRIEVAL_K`、`CANDIDATE_M` | `1200`、`300`、`50`、`3`、`2` |
| 热点问答 | `HOT_QUESTION_BM25_THRESHOLD` | `0.85` |
| 应用 | `VALID_SOURCES`、`CUSTOMER_SERVICE_PHONE`、`LOG_FILE` | `ai,java,test,ops,bigdata` 等 |

改动检索参数（切块大小等）后需要重新导入文档才会生效。

---

## 导入知识库文档

支持 PDF、Word（.docx）、PPT（.pptx）、图片（.png/.jpg/.jpeg，需 OCR）、Markdown、纯文本。

**方式一：网页上传**。进入“知识库”页，选择学科后上传文件，页面会显示入库进度；失败的任务可一键重试。

**方式二：批量导入目录**。按学科把文件放进 `rag_qa/data/<学科>_data/`（学科名取自 `VALID_SOURCES`），例如：

```text
rag_qa/data/
├── ai_data/
│   ├── LLM基础知识.pdf
│   └── RAG入门.docx
└── java_data/
    └── 集合框架.pptx
```

然后执行：

```powershell
python -m rag_qa.main --data-processing --data-dir rag_qa/data
```

不带参数运行 `python -m rag_qa.main` 会进入命令行交互问答模式。

---

## 验证与测试

```powershell
# 冒烟测试：用替身屏蔽重依赖，检查模块导入和问答编排，不需要外部服务
python tests/test_pipeline_smoke.py

# 服务健康检查（服务启动后）
curl http://127.0.0.1:8080/health

# 离线评测（需准备标注好的 JSONL，格式见 rag_qa/evaluation/README.md）
python -m rag_qa.evaluation.runner --dataset 路径\eval.jsonl --mode rag --run-name first_run
```

---

## 常见问题

**`pip install` 在 Pillow、rapidocr 等包上报错**
多半是 Python 版本过新，请使用 Python 3.11 重新创建虚拟环境。

**第一次提问很久没有回应**
正在加载 bge-m3 和 reranker 模型并连接 Milvus，观察终端日志即可；如果长时间停住，检查 `rag_qa/models/` 下的模型目录是否完整。

**提问返回“暂时无法生成回答”**
查看 `logs/app.log`。常见原因：`DASHSCOPE_API_KEY` 未填写或无效；Milvus 未启动或 `itcast` 数据库不存在；MySQL 密码错误。

**Jev 路由没有生效**
`JEV_API_KEY` 为空、调用失败或置信度低于 `JEV_CONFIDENCE_THRESHOLD` 时，系统会安全地回退到完整 RAG，因此功能上看不出报错。日志中搜索 `Jev` 可看到具体原因。使用 OpenRouter 时密钥以 `sk-or-` 开头，`JEV_ENDPOINT` 保持默认。

**知识库、运营页面显示“请求失败（500）”**
这些页面依赖 MySQL，确认 MySQL 已启动且 `.env` 中的地址和密码正确。

**端口被占用**
把启动命令中的 `--port 8080` 换成其他端口；如果同时使用 `npm run dev`，还要同步修改 `frontend/vite.config.js` 中的代理地址。
