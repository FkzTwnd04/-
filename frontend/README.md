# 智学答前端

`frontend/` 是智学答的 React 前端，问答、文档、评测和运营数据全部通过本项目的 FastAPI API 获取。

## 页面

- **问答**：创建和切换会话、发送问题、接收 WebSocket 流式回答，并提交有帮助/没帮助反馈。
- **知识库**：上传资料、查看摄取台账和状态、重新提交失败的入库任务。
- **评测**：上传 JSONL 评测集，选择 RAG 或意图模式，查看任务状态和下载最近报告。
- **运营**：查看命中层统计、脱敏后的请求日志和高频问题候选，并人工审核候选答案。

## 代码结构

- `src/App.jsx`：外壳组件，负责侧栏导航、会话列表、主题切换，以及提问、流式接收和反馈。
- `src/api.js`：`requestJson`、`postJson`、`pollUntilDone`、`streamQuestion` 等接口工具。
- `src/components.jsx`：图标、卡片、页面标题、状态标签等共用组件。
- `src/pages/`：`ChatPage`、`DocumentsPage`、`EvaluationPage`、`OperationsPage` 四个页面。
- `src/styles.css`：全部样式；设计变量在 `:root`，深色主题通过 `html[data-theme="dark"]` 覆盖变量。

## 构建和启动

在项目根目录打开 PowerShell：

```powershell
cd frontend
npm install
npm run build
cd ..
python -m uvicorn api.server:app --host 127.0.0.1 --port 8080
```

打开 `http://127.0.0.1:8080` 查看页面，`http://127.0.0.1:8080/docs` 查看 API 文档。若 Windows 没有 `python` 命令，可使用已安装的 Python 解释器执行最后一条命令。

仅开发前端时，在 `frontend/` 执行 `npm run dev`，Vite 会把 `/api` 和 `/health` 请求代理到 `http://127.0.0.1:8080`。

前端页面可在没有数据库和模型服务时启动。真正问答及知识库、评测、运营数据操作仍需按项目根目录 `.env.example` 配置相应外部服务和密钥。
