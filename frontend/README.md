# 智学答前端

`frontend/` 是智学答自己的 React 工作台。界面布局和基础样式参考并迁入 UltraRAG 的聊天工作台，页面品牌、交互和数据接口已经按本项目重写。它不连接 UltraRAG 的服务；问答、文档、评测和运营数据全部通过本项目的 FastAPI API 获取。

## 页面

- **AI 问答**：创建和切换会话、发送问题、接收 WebSocket 流式回答，并提交有帮助/没帮助反馈。
- **知识库**：上传资料、查看摄取台账和状态、重新提交失败的入库任务。
- **评测中心**：上传 JSONL 评测集，选择 RAG 或意图模式，查看任务状态和下载最近报告。
- **运营日志**：查看命中层统计、脱敏后的请求日志和高频问题候选，并人工审核候选答案。

## 构建和启动

在项目根目录打开 PowerShell：

```powershell
cd frontend
pnpm install
pnpm build
cd ..
python -m uvicorn api.server:app --host 127.0.0.1 --port 8080
```

打开 `http://127.0.0.1:8080` 查看页面，`http://127.0.0.1:8080/docs` 查看 API 文档。若 Windows 没有 `python` 命令，可使用已安装的 Python 解释器执行最后一条命令。

仅开发前端时，在 `frontend/` 执行 `pnpm dev`，Vite 会把 `/api` 和 `/health` 请求代理到 `http://127.0.0.1:8080`。

前端页面可在没有数据库和模型服务时启动。真正问答及知识库、评测、运营数据操作仍需按项目根目录 `.env.example` 配置相应外部服务和密钥。
