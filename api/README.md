# API 接入层

## 本地启动

完整的环境准备（MySQL、Redis、Milvus、模型和 `.env`）见根目录 [README.md](../README.md)。在 Windows 上先安装 Python 3.11 和 Node.js 20，再从项目根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
python -m uvicorn api.server:app --host 127.0.0.1 --port 8080
```

浏览器访问 `http://127.0.0.1:8080`。仅预览前端时，可在 `frontend/` 执行 `npm run dev`；需要在线问答、上传入库、评测和运营数据时，还要按根目录 `.env.example` 配置并启动 MySQL、Redis、Milvus 及模型服务。前端首页可在这些依赖未启动时打开。

服务日志同时输出到终端和根目录 `logs/app.log`，日志文件每 10 MiB 轮转，保留最近 5 个备份。HTTP 请求会记录 `trace_id`、路径、状态码和总耗时；WebSocket 连接及错误也会写入同一日志。

## 模块作用

`api/` 是前端和外部调用方进入系统的网络接口。它校验请求、取得共享业务对象并把结果组织成 JSON 或 WebSocket 消息；意图检索、FAQ、RAG、数据库操作分别由下层模块完成。

## 接口文件

- `server.py`：创建 FastAPI 应用，装配路由、静态页面、CORS 和 HTTP trace_id 中间件。当前 CORS 仍为开发环境宽松配置，生产环境必须按实际来源限制。
- `routes_qa.py`：提供会话创建、历史读取/清理、非流式问题入口、WebSocket 逐段回答及健康检查。答案记录的 `request_id` 在普通答案响应或 WebSocket `end` 消息中返回。
- `routes_docs.py`：接收上传文件，查询摄取任务/文档状态，并对失败任务发起重试。
- `routes_eval.py`：接收 JSONL 评测集、排入后台评测任务、读取任务状态和结果报告。
- `routes_ops.py`：提交用户评价，读取命中层统计，生成/列出 FAQ 候选，并由人工审核后回填答案。
- `deps.py`：延迟创建问答和摄取服务单例，避免各路由重复初始化模型与连接。
- `schema.py`：定义问答请求和共享数据协议；接口字段变更需同步检查静态页面和客户端。

## 核心 API 流程

```text
POST /api/query
  ├─ 可直接返回现成答案：JSON 返回答案、命中层和 request_id
  └─ 需要完整 RAG：告知客户端连接 WebSocket

WebSocket /api/stream
  → 发送 start → 持续发送 token → 发送 end（含 request_id）

POST /api/ops/feedback
  → 使用 request_id 关联回答 → 保存有帮助/没帮助评价
```

## 文档摄取和评测

文档上传由 `routes_docs.py` 调用单进程摄取服务；服务完成校验、落盘、内容哈希去重、解析切分、Milvus 写入和状态更新。当前任务调度和状态缓存不是共享持久队列，跨进程扩容及服务重启恢复能力有限。

评测路由调用 `rag_qa/evaluation/runner.py`。客户端上传的数据必须符合模块 README 中的 JSONL 格式；报告保存评测问句和模型答案，应妥善控制访问和保留时间。

## 安全和部署边界

目前没有身份认证或授权，服务按默认单租户运行。不能把请求体里的任意 `tenant_id` 当作可信租户身份。部署前应补充认证、权限、限流、上传类型/大小限制、CORS 来源限制以及错误响应脱敏；同步数据库和模型调用也需要移出高并发异步请求的阻塞路径。
