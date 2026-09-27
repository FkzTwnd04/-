/**
 * 智学答前端外壳：左侧导航与会话列表、顶部状态栏，以及问答 / 知识库 / 评测 / 运营四个页面的切换。
 * 会话与消息保存在浏览器 localStorage，所有数据请求都发往本项目的 FastAPI。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { clockTime, postJson, requestJson, streamQuestion } from "./api.js";
import { Icon } from "./components.jsx";
import ChatPage from "./pages/ChatPage.jsx";
import DocumentsPage from "./pages/DocumentsPage.jsx";
import EvaluationPage from "./pages/EvaluationPage.jsx";
import OperationsPage from "./pages/OperationsPage.jsx";

const STORE_KEY = "zhixueda.sessions";
const THEME_KEY = "zhixueda.theme";

const SECTIONS = [
  { id: "chat", label: "问答", icon: "chat", hint: "基于学习资料检索与回答" },
  { id: "documents", label: "知识库", icon: "library", hint: "管理入库的学习资料" },
  { id: "evaluation", label: "评测", icon: "gauge", hint: "离线检验检索与回答质量" },
  { id: "operations", label: "运营", icon: "pulse", hint: "请求统计与高频问题审核" },
];

function loadSessions() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORE_KEY) || "[]");
    return Array.isArray(stored) ? stored : [];
  } catch {
    return [];
  }
}

function titleFrom(messages) {
  return messages.find((item) => item.role === "user")?.text?.slice(0, 24) || "新对话";
}

export default function App() {
  const [section, setSection] = useState("chat");
  const [sessions, setSessions] = useState(loadSessions);
  const [currentId, setCurrentId] = useState("");
  const [busy, setBusy] = useState(false);
  const [sources, setSources] = useState([]);
  const [source, setSource] = useState("");
  const [online, setOnline] = useState(false);
  const [notice, setNotice] = useState("");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_KEY) || "light");

  const current = sessions.find((item) => item.id === currentId);
  const messages = current?.messages || [];

  useEffect(() => { localStorage.setItem(STORE_KEY, JSON.stringify(sessions)); }, [sessions]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const updateMessages = useCallback((sessionId, change) => {
    setSessions((list) => list
      .map((item) => {
        if (item.id !== sessionId) return item;
        const next = typeof change === "function" ? change(item.messages || []) : change;
        return { ...item, messages: next, title: item.title && item.title !== "新对话" ? item.title : titleFrom(next), updatedAt: Date.now() };
      })
      .sort((a, b) => b.updatedAt - a.updatedAt));
  }, []);

  const startSession = useCallback(async () => {
    const created = await requestJson("/api/create_session", { method: "POST" });
    const session = { id: created.session_id, title: "新对话", messages: [], updatedAt: Date.now() };
    setSessions((list) => [session, ...list]);
    setCurrentId(session.id);
    return session.id;
  }, []);

  async function newSession() {
    try {
      await startSession();
      setSection("chat");
      setDrawerOpen(false);
      setNotice("");
    } catch (issue) {
      setNotice(`创建会话失败：${issue.message}`);
    }
  }

  const booted = useRef(false);
  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    requestJson("/api/system/status").then((data) => setOnline(data.status === "running")).catch(() => setOnline(false));
    requestJson("/api/sources").then((data) => setSources(data.sources || [])).catch(() => setSources([]));
    const stored = loadSessions();
    if (stored.length) setCurrentId(stored[0].id);
    else startSession().catch((issue) => setNotice(`创建会话失败：${issue.message}`));
  }, [startSession]);

  async function openSession(session) {
    setCurrentId(session.id);
    setSection("chat");
    setDrawerOpen(false);
    if (session.messages?.length) return;
    try {
      const data = await requestJson(`/api/history/${encodeURIComponent(session.id)}`);
      const restored = (data.history || []).flatMap((turn) => [
        { role: "user", text: turn.question },
        { role: "assistant", text: turn.answer },
      ]);
      if (restored.length) updateMessages(session.id, restored);
    } catch {
      // 历史接口依赖数据库；不可用时仍显示本地保存的消息。
    }
  }

  function removeSession(sessionId) {
    setSessions((list) => list.filter((item) => item.id !== sessionId));
    if (sessionId === currentId) {
      setCurrentId("");
      startSession().catch((issue) => setNotice(`创建会话失败：${issue.message}`));
    }
  }

  async function ask(question) {
    let sessionId = currentId;
    if (!sessionId) {
      try {
        sessionId = await startSession();
      } catch (issue) {
        setNotice(`创建会话失败：${issue.message}`);
        return;
      }
    }
    const answerId = `answer-${Date.now()}`;
    const time = clockTime();
    updateMessages(sessionId, (list) => [
      ...list,
      { role: "user", text: question, time },
      { role: "assistant", id: answerId, text: "", pending: true, time },
    ]);
    const patchAnswer = (fields) => updateMessages(sessionId, (list) => list.map((item) => (
      item.id === answerId ? { ...item, pending: false, ...fields } : item
    )));

    setBusy(true);
    setNotice("");
    const payload = { query: question, source_filter: source || null, session_id: sessionId };
    try {
      const reply = await postJson("/api/query", payload);
      if (!reply.is_streaming) {
        patchAnswer({ text: reply.answer || "", requestId: reply.request_id, answerMode: reply.answer_mode });
        return;
      }
      const final = await streamQuestion(payload, (text) => patchAnswer({ text }));
      patchAnswer({ text: final.text, requestId: final.request_id, answerMode: final.answer_mode, citations: final.citations || [] });
    } catch {
      patchAnswer({ text: "暂时无法生成回答，请检查模型服务和 API 密钥配置后重试。", failed: true });
      setNotice("请求失败，详细原因见 logs/app.log；若是鉴权错误，请检查 .env 中的模型 API 密钥。");
    } finally {
      setBusy(false);
    }
  }

  async function rate(requestId, rating) {
    try {
      await postJson("/api/ops/feedback", { request_id: requestId, rating });
      updateMessages(currentId, (list) => list.map((item) => (
        item.requestId === requestId ? { ...item, feedback: rating === 1 ? "感谢反馈" : "已记录" } : item
      )));
    } catch (issue) {
      setNotice(`反馈未保存：${issue.message}`);
    }
  }

  function go(id) {
    setSection(id);
    setDrawerOpen(false);
    setNotice("");
  }

  const active = SECTIONS.find((item) => item.id === section) || SECTIONS[0];

  return (
    <div className="shell">
      {drawerOpen ? <button type="button" className="scrim" aria-label="关闭菜单" onClick={() => setDrawerOpen(false)} /> : null}

      <aside className={`rail ${drawerOpen ? "rail-open" : ""}`}>
        <div className="brand">
          <span className="brand-seal">知</span>
          <div>
            <strong>智学答</strong>
            <small>学习资料问答</small>
          </div>
        </div>

        <button type="button" className="rail-new" onClick={() => void newSession()}>
          <Icon name="plus" size={16} />新对话
        </button>

        <nav className="rail-nav" aria-label="功能导航">
          {SECTIONS.map((item) => (
            <button type="button" key={item.id} className={item.id === section ? "is-active" : ""} onClick={() => go(item.id)}>
              <Icon name={item.icon} size={17} />{item.label}
            </button>
          ))}
        </nav>

        <div className="rail-sessions">
          <div className="rail-caption">
            <span>历史对话</span><span>{sessions.length}</span>
          </div>
          {sessions.length ? (
            <ul>
              {sessions.map((item) => (
                <li key={item.id} className={item.id === currentId && section === "chat" ? "is-active" : ""}>
                  <button type="button" className="session-open" onClick={() => void openSession(item)}>{item.title || "新对话"}</button>
                  <button type="button" className="session-drop" title="删除对话" onClick={() => removeSession(item.id)}><Icon name="x" size={13} /></button>
                </li>
              ))}
            </ul>
          ) : <p className="rail-empty">还没有对话</p>}
        </div>

        <footer className="rail-foot">
          <span className={`dot ${online ? "dot-on" : ""}`} />
          {online ? "后端已连接" : "后端未连接"}
        </footer>
      </aside>

      <main className="stage">
        <header className="stage-bar">
          <button type="button" className="icon-btn menu-btn" aria-label="打开菜单" onClick={() => setDrawerOpen(true)}><Icon name="menu" /></button>
          <div className="stage-title">
            <strong>{active.label}</strong>
            <span>{active.hint}</span>
          </div>
          <button type="button" className="icon-btn" title="切换主题" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            <Icon name={theme === "dark" ? "sun" : "moon"} />
          </button>
        </header>

        {notice ? (
          <div className="notice" role="status">
            <span>{notice}</span>
            <button type="button" aria-label="关闭提示" onClick={() => setNotice("")}><Icon name="x" size={14} /></button>
          </div>
        ) : null}

        <div className="stage-body">
          {section === "chat" ? (
            <ChatPage messages={messages} busy={busy} sources={sources} source={source}
              onSourceChange={setSource} onSend={(question) => void ask(question)} onFeedback={rate} />
          ) : null}
          {section === "documents" ? <DocumentsPage sources={sources} notify={setNotice} /> : null}
          {section === "evaluation" ? <EvaluationPage notify={setNotice} /> : null}
          {section === "operations" ? <OperationsPage notify={setNotice} /> : null}
        </div>
      </main>
    </div>
  );
}
