/**
 * 智学答的浏览器工作台入口：用 UltraRAG 前端的侧栏、会话列表、中央聊天区和底部输入区布局，
 * 重新实现本项目的问答、知识文档、离线评测和运营日志页面。所有网络请求均调用本项目 FastAPI，
 * 会话展示状态保存在浏览器本地；外部数据库、向量库和模型只在用户实际使用对应功能时由后端连接。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const SESSION_KEY = "zhixueda.chat.sessions.v1";
const PAGE_ITEMS = [
  { id: "chat", label: "AI 问答", icon: "chat" },
  { id: "documents", label: "知识库", icon: "book" },
  { id: "evaluation", label: "评测中心", icon: "chart" },
  { id: "operations", label: "运营日志", icon: "activity" },
];

function readSessions() {
  try {
    const value = JSON.parse(localStorage.getItem(SESSION_KEY) || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const result = contentType.includes("application/json") ? await response.json() : {};
  if (!response.ok) {
    throw new Error(result.detail || result.error || `请求失败（${response.status}）`);
  }
  return result;
}

function Icon({ name, size = 18 }) {
  const shapes = {
    chat: <><path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H5l-1-4.5A7.5 7.5 0 1 1 20 11.5Z"/><path d="M8 10h8M8 14h5"/></>,
    book: <><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 0 4 21z"/><path d="M4 5.5v13A2.5 2.5 0 0 1 6.5 16H20M8 7h8"/></>,
    chart: <><path d="M4 19V5M4 19h17"/><path d="m7 15 4-4 3 2 5-6"/></>,
    activity: <><path d="M3 12h4l3-8 4 16 3-8h4"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h8"/></>,
    upload: <><path d="M12 16V4m0 0L7 9m5-5 5 5"/><path d="M4 16v4h16v-4"/></>,
    sun: <><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42"/></>,
    menu: <><path d="M4 6h16M4 12h16M4 18h16"/></>,
    trash: <><path d="M3 6h18M8 6V4h8v2m3 0-1 14H6L5 6m4 4v6m6-6v6"/></>,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.5 9A7 7 0 0 1 18 6l2 2M4 16l2 2a7 7 0 0 0 12.5-3"/></>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>,
    sunMoon: <><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4 8.5 8.5 0 1 0 20 14.5Z"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{shapes[name] || shapes.chat}</svg>;
}

function escapeHtml(text) {
  return String(text || "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function renderAnswer(text) {
  const safe = escapeHtml(text);
  return safe
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/\n/g, "<br>");
}

function EmptyState({ onPrompt }) {
  const examples = [
    "帮我梳理 RAG 问答系统的核心流程",
    "BM25 和向量检索分别适合解决什么问题？",
    "怎样评估检索结果是否相关？",
  ];
  return (
    <div className="react-chat-empty zh-empty">
      <div className="zh-empty-mark">知</div>
      <p className="zh-eyebrow">智学答 · 学习知识工作台</p>
      <h1>你好，今天想了解什么？</h1>
      <p className="zh-empty-copy">围绕已接入的学习资料提问，我会先检索依据，再组织答案。</p>
      <div className="zh-prompt-grid">
        {examples.map((example) => (
          <button className="zh-prompt-card" key={example} onClick={() => onPrompt(example)}>
            <span>{example}</span><span className="zh-prompt-arrow">↗</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function ChatMessage({ message, onFeedback }) {
  const assistant = message.role === "assistant";
  const citedFiles = [...new Set((message.citations || []).map((citation) => citation.filename).filter(Boolean))];
  return (
    <article className={`zh-message ${assistant ? "assistant" : "user"}`}>
      <div className={`zh-avatar ${assistant ? "assistant" : "user"}`}>{assistant ? "知" : "我"}</div>
      <div className="zh-message-main">
        <div className="zh-message-head"><strong>{assistant ? "智学答" : "我"}</strong><span>{message.time || ""}</span></div>
        <div className={`react-chat-bubble ${assistant ? "assistant" : "user"}`}>
          {message.pending ? <span className="zh-thinking"><i/><i/><i/> 正在检索并整理答案</span> : (
            <div className="react-chat-bubble-content react-chat-markdown" dangerouslySetInnerHTML={{ __html: renderAnswer(message.text) }} />
          )}
        </div>
        {assistant && !message.pending && message.requestId ? (
          <div className="zh-feedback-row">
            {citedFiles.length ? <span>知识库文件：{citedFiles.join("、")}</span> : message.answerMode === "database" ? <span>回答来源：数据库热点问答</span> : message.answerMode === "rag" ? <span>本次未检索到知识库文件</span> : message.answerMode === "general_llm" ? null : <span>这条回答对你有帮助吗？</span>}
            <button title="有帮助" onClick={() => onFeedback(message.requestId, 1)}>👍</button>
            <button title="没帮助" onClick={() => onFeedback(message.requestId, -1)}>👎</button>
            {message.feedback ? <em>{message.feedback}</em> : null}
          </div>
        ) : null}
      </div>
    </article>
  );
}

function Card({ children, className = "" }) {
  return <section className={`zh-card ${className}`}>{children}</section>;
}

export default function App() {
  const [page, setPage] = useState("chat");
  const [sessions, setSessions] = useState(readSessions);
  const [activeSession, setActiveSession] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [source, setSource] = useState("");
  const [sources, setSources] = useState([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [status, setStatus] = useState(null);
  const [dark, setDark] = useState(() => localStorage.getItem("zhixueda.theme") === "dark");
  const [mobileSidebar, setMobileSidebar] = useState(false);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => { localStorage.setItem(SESSION_KEY, JSON.stringify(sessions)); }, [sessions]);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("zhixueda.theme", dark ? "dark" : "light");
  }, [dark]);
  useEffect(() => {
    requestJson("/api/system/status").then(setStatus).catch(() => setStatus({ status: "offline" }));
    requestJson("/api/sources").then((data) => setSources(data.sources || [])).catch(() => setSources([]));
    if (!sessions.length) void createNewSession();
    else {
      const recent = sessions[0];
      setActiveSession(recent.id);
      setMessages(recent.messages || []);
    }
  }, []);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, busy]);

  const saveMessages = useCallback((sessionId, nextMessages) => {
    setSessions((previous) => previous.map((session) => session.id === sessionId
      ? { ...session, messages: nextMessages, title: session.title || nextMessages.find((item) => item.role === "user")?.text?.slice(0, 24) || "新对话", updatedAt: Date.now() }
      : session).sort((a, b) => b.updatedAt - a.updatedAt));
  }, []);

  async function createNewSession() {
    try {
      const data = await requestJson("/api/create_session", { method: "POST" });
      const session = { id: data.session_id, title: "新对话", messages: [], updatedAt: Date.now() };
      setSessions((previous) => [session, ...previous]);
      setActiveSession(session.id);
      setMessages([]);
      setPage("chat");
      setMobileSidebar(false);
      setNotice("");
    } catch (error) { setNotice(`创建会话失败：${error.message}`); }
  }

  async function openSession(session) {
    setActiveSession(session.id);
    setMessages(session.messages || []);
    setPage("chat");
    setMobileSidebar(false);
    if (session.messages?.length) return;
    try {
      const data = await requestJson(`/api/history/${encodeURIComponent(session.id)}`);
      const restored = (data.history || []).flatMap((item) => [
        { role: "user", text: item.question, time: "" },
        { role: "assistant", text: item.answer, time: "" },
      ]);
      setMessages(restored);
      saveMessages(session.id, restored);
    } catch { /* 会话本地消息仍可用；初次预览 UI 不要求数据库在线。 */ }
  }

  async function sendQuestion(rawText = input) {
    const question = rawText.trim();
    if (!question || busy) return;
    let sessionId = activeSession;
    if (!sessionId) {
      try {
        const created = await requestJson("/api/create_session", { method: "POST" });
        sessionId = created.session_id;
        setActiveSession(sessionId);
        setSessions((previous) => [{ id: sessionId, title: "新对话", messages: [], updatedAt: Date.now() }, ...previous]);
      } catch (error) { setNotice(`创建会话失败：${error.message}`); return; }
    }
    const userMessage = { role: "user", text: question, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) };
    const answerId = `answer-${Date.now()}`;
    const pendingAnswer = { role: "assistant", id: answerId, text: "", pending: true, time: userMessage.time };
    const startingMessages = [...messages, userMessage, pendingAnswer];
    setMessages(startingMessages);
    saveMessages(sessionId, startingMessages);
    setInput("");
    setBusy(true);
    setNotice("");

    const replacePending = (updates) => setMessages((previous) => {
      const next = previous.map((message) => message.id === answerId ? { ...message, pending: false, ...updates } : message);
      saveMessages(sessionId, next);
      return next;
    });

    try {
      const data = await requestJson("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question, source_filter: source || null, session_id: sessionId }),
      });
      if (!data.is_streaming) {
        replacePending({ text: data.answer || "", requestId: data.request_id, hitLayer: data.hit_layer, answerMode: data.answer_mode });
        return;
      }
      await streamAnswer(question, sessionId, answerId, replacePending);
    } catch (error) {
      replacePending({ text: "暂时无法生成回答，请检查模型服务和 API 密钥配置后重试。", failed: true });
      setNotice("请求失败。具体原因请查看项目 logs/app.log；如果是模型鉴权错误，请检查 .env 中的模型 API 密钥。");
    } finally { setBusy(false); }
  }

  function streamAnswer(question, sessionId, answerId, replacePending) {
    return new Promise((resolve, reject) => {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const socket = new WebSocket(`${protocol}//${location.host}/api/stream`);
      let answer = "";
      socket.onopen = () => socket.send(JSON.stringify({ query: question, source_filter: source || null, session_id: sessionId }));
      socket.onmessage = (event) => {
        let data;
        try { data = JSON.parse(event.data); } catch { return; }
        if (data.type === "token") {
          answer += data.token || "";
          replacePending({ text: answer });
        } else if (data.type === "end") {
          replacePending({ text: answer, requestId: data.request_id, answerMode: data.answer_mode, citations: data.citations || [] });
          socket.close();
          resolve();
        } else if (data.type === "error") {
          socket.close();
          reject(new Error(data.error || "流式回答失败"));
        }
      };
      socket.onerror = () => reject(new Error("WebSocket 连接失败"));
      socket.onclose = () => resolve();
    });
  }

  async function sendFeedback(requestId, rating) {
    try {
      await requestJson("/api/ops/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: requestId, rating }),
      });
      setMessages((previous) => {
        const next = previous.map((message) => message.requestId === requestId ? { ...message, feedback: rating === 1 ? "感谢反馈" : "已记录" } : message);
        if (activeSession) saveMessages(activeSession, next);
        return next;
      });
    } catch (error) { setNotice(`反馈未保存：${error.message}`); }
  }

  function removeSession(sessionId, event) {
    event.stopPropagation();
    setSessions((previous) => previous.filter((session) => session.id !== sessionId));
    if (activeSession === sessionId) {
      setActiveSession(""); setMessages([]); void createNewSession();
    }
  }

  function changePage(nextPage) {
    setPage(nextPage); setMobileSidebar(false); setNotice("");
  }

  const activeTitle = useMemo(() => PAGE_ITEMS.find((item) => item.id === page)?.label || "AI 问答", [page]);
  const submitOnEnter = (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendQuestion(); }
  };

  return (
    <div className={`react-chat-page zh-app ${dark ? "is-dark" : ""}`}>
      {mobileSidebar ? <button className="zh-mobile-backdrop" aria-label="关闭导航" onClick={() => setMobileSidebar(false)} /> : null}
      <aside className={`react-chat-sidebar zh-sidebar ${mobileSidebar ? "is-open" : ""}`}>
        <div className="react-chat-sidebar-header zh-sidebar-header">
          <button className="zh-brand" onClick={() => changePage("chat")}>
            <span className="zh-brand-mark">知</span>
            <span><strong>智学答</strong><small>学习知识工作台</small></span>
          </button>
          <button className="zh-new-chat" onClick={() => void createNewSession()}><Icon name="plus" size={17}/> 新建对话</button>
        </div>
        <nav className="react-chat-view-nav zh-nav" aria-label="工作台导航">
          {PAGE_ITEMS.map((item) => (
            <button key={item.id} className={page === item.id ? "active" : ""} onClick={() => changePage(item.id)}>
              <Icon name={item.icon}/><span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="react-chat-session-section zh-session-section">
          <div className="react-chat-session-head"><span>最近对话</span><span>{sessions.length}</span></div>
          <div className="react-chat-session-list">
            {sessions.length ? sessions.map((session) => (
              <button key={session.id} className={`react-chat-session-item ${activeSession === session.id && page === "chat" ? "active" : ""}`} onClick={() => void openSession(session)}>
                <span>{session.title || "新对话"}</span>
                <span className="react-chat-session-actions"><span className="zh-session-remove" role="button" title="删除会话" onClick={(event) => removeSession(session.id, event)}><Icon name="close" size={14}/></span></span>
              </button>
            )) : <div className="react-chat-side-placeholder"><p>还没有对话</p><span className="react-chat-side-subtle">新建对话后会显示在这里</span></div>}
          </div>
        </div>
        <div className="react-chat-sidebar-footer zh-sidebar-footer">
          <div className="zh-sidebar-status"><span className={`zh-status-dot ${status?.status === "running" ? "online" : ""}`}/>{status?.status === "running" ? "服务已启动" : "等待连接后端"}</div>
          <button className="zh-theme-toggle" onClick={() => setDark((value) => !value)}><Icon name={dark ? "sun" : "sunMoon"} size={16}/>{dark ? "浅色模式" : "深色模式"}</button>
        </div>
      </aside>

      <main className="react-chat-main zh-main">
        <header className="react-chat-topbar zh-topbar">
          <button className="react-chat-mobile-menu" aria-label="打开导航" onClick={() => setMobileSidebar(true)}><Icon name="menu"/></button>
          <div className="zh-top-title"><strong>{activeTitle}</strong><span>{page === "chat" ? "基于学习资料检索与回答" : "智学答系统管理"}</span></div>
          <div className="zh-top-actions">
            <span className={`zh-service-pill ${status?.status === "running" ? "ready" : ""}`}><i/>{status?.status === "running" ? "服务在线" : "服务未连接"}</span>
            <button className="zh-icon-button" title="切换主题" onClick={() => setDark((value) => !value)}><Icon name={dark ? "sun" : "sunMoon"}/></button>
          </div>
        </header>

        {notice ? <div className="zh-notice" role="status"><span>{notice}</span><button onClick={() => setNotice("")}><Icon name="close" size={15}/></button></div> : null}

        {page === "chat" ? (
          <div className="zh-chat-workspace">
            <div className="react-chat-history zh-chat-history">
              {messages.length ? messages.map((message, index) => <ChatMessage key={message.id || `${message.role}-${index}`} message={message} onFeedback={sendFeedback}/>) : <EmptyState onPrompt={(text) => { setInput(text); inputRef.current?.focus(); }}/ >}
              <div ref={chatEndRef}/>
            </div>
            <div className="zh-composer-wrap">
              <div className="zh-composer">
                <div className="react-chat-composer-left zh-composer-left">
                  <label className="zh-source-select"><span>知识来源</span><select value={source} onChange={(event) => setSource(event.target.value)}><option value="">全部资料</option>{sources.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
                  <span className="zh-composer-hint">Enter 发送 · Shift + Enter 换行</span>
                </div>
                <div className="zh-input-row">
                  <textarea ref={inputRef} className="react-chat-input zh-input" value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={submitOnEnter} placeholder="输入问题，按 Enter 发送…" rows={1} disabled={busy}/>
                  <button className="zh-send-button" title="发送问题" onClick={() => void sendQuestion()} disabled={busy || !input.trim()}>{busy ? <span className="zh-send-spinner"/> : <Icon name="send" size={18}/>}</button>
                </div>
              </div>
              <div className="zh-disclaimer">智学答可能会出错，请结合资料来源判断重要信息。</div>
            </div>
          </div>
        ) : null}
        {page === "documents" ? <DocumentsPage sources={sources} showNotice={setNotice}/> : null}
        {page === "evaluation" ? <EvaluationPage showNotice={setNotice}/> : null}
        {page === "operations" ? <OperationsPage showNotice={setNotice}/> : null}
      </main>
    </div>
  );
}

function DocumentsPage({ sources, showNotice }) {
  const [files, setFiles] = useState([]);
  const [upload, setUpload] = useState(null);
  const [selectedSource, setSelectedSource] = useState(sources[0] || "ai");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef(null);
  const refresh = useCallback(async () => {
    setLoading(true); setError("");
    try { const result = await requestJson("/api/docs?limit=100"); setFiles(result.documents || []); }
    catch (issue) { setError(issue.message); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);

  async function uploadFile(file) {
    if (!file) return;
    setUpload({ filename: file.name, status: "上传中" }); setError("");
    const data = new FormData(); data.append("file", file); data.append("source", selectedSource || "ai");
    try {
      const task = await requestJson("/api/docs/upload", { method: "POST", body: data });
      setUpload({ filename: file.name, status: "正在入库", taskId: task.task_id });
      await pollUntilDone(`/api/docs/tasks/${task.task_id}`, (result) => {
        setUpload({ filename: file.name, status: result.status, error: result.error });
        if (["completed", "failed", "ingested", "duplicate"].includes(result.status)) return true;
        return false;
      });
      await refresh();
    } catch (issue) { setUpload({ filename: file.name, status: "失败", error: issue.message }); setError(issue.message); }
  }

  async function retryFailed() {
    try { const result = await requestJson("/api/docs/retry", { method: "POST" }); showNotice(`已重新提交 ${result.submitted || 0} 个失败任务。`); await refresh(); }
    catch (issue) { setError(issue.message); }
  }

  return (
    <div className="zh-page-content">
      <div className="zh-page-heading"><div><p className="zh-eyebrow">知识管理</p><h1>文档知识库</h1><p>上传学习资料后，系统会解析内容、切分文本并写入向量库。</p></div><button className="zh-secondary-button" onClick={() => void refresh()}><Icon name="refresh" size={16}/> 刷新状态</button></div>
      <Card className="zh-upload-card">
        <div className="zh-card-title"><div className="zh-card-icon"><Icon name="upload"/></div><div><h2>导入文档</h2><p>支持 PDF、Word、PPT 和系统已配置的其他格式</p></div></div>
        <div className="zh-upload-controls"><label>知识来源<select value={selectedSource} onChange={(event) => setSelectedSource(event.target.value)}>{(sources.length ? sources : ["ai"]).map((item) => <option key={item}>{item}</option>)}</select></label><input ref={fileInput} type="file" hidden onChange={(event) => { void uploadFile(event.target.files?.[0]); event.target.value = ""; }}/><button className="zh-primary-button" onClick={() => fileInput.current?.click()}><Icon name="upload" size={16}/> 选择文件并上传</button></div>
        {upload ? <div className="zh-upload-progress"><span>{upload.filename}</span><strong>{upload.status}</strong>{upload.error ? <small>{upload.error}</small> : null}</div> : null}
      </Card>
      <Card>
        <div className="zh-section-heading"><div><h2>已登记文档</h2><p>这里展示摄取台账中的文档和最近处理状态。</p></div><button className="zh-secondary-button" onClick={() => void retryFailed()}>重试失败任务</button></div>
        {error ? <div className="zh-inline-error">{error}</div> : null}
        {loading ? <div className="zh-empty-table">正在读取文档状态…</div> : files.length ? <div className="zh-table-wrap"><table className="zh-table"><thead><tr><th>文档</th><th>状态</th><th>文本块</th><th>重试</th><th>更新时间</th></tr></thead><tbody>{files.map((file) => <tr key={file.doc_id}><td><strong>{file.title || file.source_path}</strong><small>{file.source_path}</small></td><td><span className={`zh-status-tag ${file.status}`}>{file.status}</span>{file.error_msg ? <small className="zh-error-text">{file.error_msg}</small> : null}</td><td>{file.chunk_count || 0}</td><td>{file.retry_count || 0}</td><td>{formatDate(file.updated_at)}</td></tr>)}</tbody></table></div> : <div className="zh-empty-table">暂无文档。上传一个资料文件后，它会显示在这里。</div>}
      </Card>
    </div>
  );
}

function EvaluationPage({ showNotice }) {
  const [mode, setMode] = useState("rag");
  const [runName, setRunName] = useState("智学答评测");
  const [file, setFile] = useState(null);
  const [task, setTask] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function startRun(event) {
    event.preventDefault();
    if (!file) { setError("请先选择 JSONL 评测集。"); return; }
    setBusy(true); setError(""); setTask(null);
    const body = new FormData(); body.append("file", file); body.append("mode", mode); body.append("run_name", runName);
    try {
      const started = await requestJson("/api/eval/runs", { method: "POST", body });
      setTask({ job_id: started.job_id, status: started.status, mode: started.mode });
      await pollUntilDone(`/api/eval/runs/${started.job_id}`, (result) => {
        setTask(result);
        if (["completed", "failed"].includes(result.status)) return true;
        return false;
      });
    } catch (issue) { setError(issue.message); }
    finally { setBusy(false); }
  }

  async function downloadLatest() {
    try {
      const report = await requestJson("/api/eval/latest");
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
      anchor.href = url; anchor.download = "zhixueda-latest-evaluation.json"; anchor.click(); URL.revokeObjectURL(url);
    } catch (issue) { showNotice(issue.message); }
  }

  return (
    <div className="zh-page-content">
      <div className="zh-page-heading"><div><p className="zh-eyebrow">质量验证</p><h1>离线评测</h1><p>上传人工标注的 JSONL 数据，运行 RAG 回答或意图识别评测。</p></div><button className="zh-secondary-button" onClick={() => void downloadLatest()}>下载最近报告</button></div>
      <Card className="zh-form-card">
        <form onSubmit={startRun}>
          <div className="zh-form-grid"><label>评测模式<select value={mode} onChange={(event) => setMode(event.target.value)}><option value="rag">RAG 问答与检索</option><option value="intent">意图识别</option></select></label><label>报告名称<input value={runName} onChange={(event) => setRunName(event.target.value)} maxLength={60}/></label></div>
          <label className="zh-file-picker"><Icon name="file"/><span>{file ? file.name : "选择 JSONL 文件"}</span><input type="file" accept=".jsonl,application/jsonl" onChange={(event) => setFile(event.target.files?.[0] || null)}/></label>
          <button className="zh-primary-button" type="submit" disabled={busy}>{busy ? "评测运行中…" : "开始评测"}</button>
        </form>
        <div className="zh-format-help"><strong>数据格式要求</strong>{mode === "rag" ? <code>{'{"id":"样本编号","query":"问题","reference_answer":"人工核对答案","relevant_chunk_ids":["真实 Chunk ID"]}'}</code> : <code>{'{"id":"样本编号","query":"问题","expected_intent":"人工确认意图"}'}</code>}<p>每行一个 JSON 对象。检索评测中的 Chunk ID 必须来自当前知识库；不填写时会跳过检索质量指标。</p></div>
      </Card>
      {error ? <div className="zh-inline-error">{error}</div> : null}
      {task ? <Card className="zh-task-card"><div><span className={`zh-status-tag ${task.status}`}>{task.status}</span><strong>{task.run_name || task.job_id}</strong></div>{task.error ? <p className="zh-error-text">{task.error}</p> : null}{task.status === "completed" ? <pre>{JSON.stringify(task.result || task, null, 2)}</pre> : null}</Card> : null}
    </div>
  );
}

function OperationsPage({ showNotice }) {
  const [stats, setStats] = useState(null);
  const [logs, setLogs] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true); setError("");
    try {
      const [statistics, requests, pending] = await Promise.all([
        requestJson("/api/ops/stats?days=7"),
        requestJson("/api/ops/requests?limit=50"),
        requestJson("/api/ops/candidates?status=pending&limit=100"),
      ]);
      setStats(statistics.layers || {}); setLogs(requests.items || []); setCandidates(pending.items || []);
    } catch (issue) { setError(issue.message); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);

  async function buildCandidates() {
    try { const result = await requestJson("/api/ops/candidates/build?min_freq=3&days=7", { method: "POST" }); showNotice(`发现并登记 ${result.created_or_updated} 个候选问题。`); await refresh(); }
    catch (issue) { setError(issue.message); }
  }

  async function review(candidate, approved) {
    const answer = approved ? window.prompt(`为“${candidate.cluster_question}”填写人工确认的答案：`) : "";
    if (approved && !answer?.trim()) return;
    try {
      await requestJson(`/api/ops/candidates/${candidate.id}/review`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved, answer: answer || "" }),
      });
      await refresh();
    } catch (issue) { setError(issue.message); }
  }

  const metricRows = Object.entries(stats || {});
  return (
    <div className="zh-page-content">
      <div className="zh-page-heading"><div><p className="zh-eyebrow">系统观测</p><h1>运营与请求日志</h1><p>查看答案复用层、请求耗时、用户反馈和需要人工补充的高频问题。</p></div><button className="zh-secondary-button" onClick={() => void refresh()}><Icon name="refresh" size={16}/> 刷新</button></div>
      {error ? <div className="zh-inline-error">{error}。请确认后端服务及 MySQL 已启动。</div> : null}
      <div className="zh-metric-grid">{metricRows.length ? metricRows.map(([layer, values]) => <Card key={layer} className="zh-metric-card"><span>{layer}</span><strong>{values.requests}</strong><small>请求 · {Math.round(values.share * 100)}% 占比</small><small>平均 {Math.round(values.average_latency_ms)} ms · 点踩 {Math.round(values.thumbs_down_rate * 100)}%</small></Card>) : <Card className="zh-metric-card"><span>最近 7 天</span><strong>—</strong><small>{loading ? "正在读取统计" : "暂无请求记录"}</small></Card>}</div>
      <Card><div className="zh-section-heading"><div><h2>待审核知识候选</h2><p>候选由高频 L3 问题汇总，不会自动生成或发布答案。</p></div><button className="zh-primary-button" onClick={() => void buildCandidates()}>生成候选</button></div>
        {candidates.length ? <div className="zh-candidate-list">{candidates.map((candidate) => <div className="zh-candidate" key={candidate.id}><div><strong>{candidate.cluster_question}</strong><small>出现 {candidate.frequency} 次 · {candidate.sample_queries?.join(" / ")}</small></div><div><button className="zh-secondary-button" onClick={() => void review(candidate, false)}>忽略</button><button className="zh-primary-button" onClick={() => void review(candidate, true)}>审核并补答案</button></div></div>)}</div> : <div className="zh-empty-table">{loading ? "正在读取候选…" : "当前没有待审核问题。"}</div>}
      </Card>
      <Card><div className="zh-section-heading"><div><h2>最近问答请求</h2><p>问题和回答在写入运营日志前会进行手机号、邮箱和身份证号脱敏。</p></div><span className="zh-count-label">{logs.length} 条</span></div>
        {logs.length ? <div className="zh-table-wrap"><table className="zh-table zh-log-table"><thead><tr><th>问题 / 回答摘要</th><th>命中层</th><th>意图</th><th>耗时</th><th>评价</th><th>时间</th></tr></thead><tbody>{logs.map((log) => <tr key={log.request_id}><td><strong>{log.query}</strong><small>{log.answer}</small></td><td><span className="zh-status-tag">{log.hit_layer}</span></td><td>{log.intent || "—"}</td><td>{log.latency_ms} ms</td><td>{log.rating === 1 ? "👍" : log.rating === -1 ? "👎" : "—"}</td><td>{formatDate(log.created_at)}</td></tr>)}</tbody></table></div> : <div className="zh-empty-table">{loading ? "正在读取日志…" : "暂无请求日志。"}</div>}
      </Card>
    </div>
  );
}

async function pollUntilDone(url, onResult) {
  for (let attempt = 0; attempt < 600; attempt += 1) {
    const result = await requestJson(url);
    if (onResult(result)) return result;
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
  throw new Error("等待任务超时，请稍后刷新状态。");
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}
