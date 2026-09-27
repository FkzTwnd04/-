/**
 * 问答页：展示当前会话的消息流、示例问题和输入区。发送、反馈等动作由 App 传入的回调完成。
 */
import { useEffect, useRef, useState } from "react";
import { Icon } from "../components.jsx";

const STARTER_QUESTIONS = [
  "帮我梳理 RAG 问答系统的核心流程",
  "BM25 和向量检索分别适合解决什么问题？",
  "怎样评估检索结果是否相关？",
];

const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

/** 先整体转义再做少量 Markdown 替换，模型输出里的 HTML 不会被执行。 */
function answerToHtml(text) {
  return String(text || "")
    .replace(/[&<>"']/g, (ch) => HTML_ESCAPES[ch])
    .split("\n")
    .map((line) => {
      const heading = line.match(/^(#{1,3})\s+(.*)$/);
      const body = heading ? heading[2] : line;
      const inline = body
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      return heading ? `<h${heading[1].length + 2}>${inline}</h${heading[1].length + 2}>` : inline;
    })
    .join("<br>")
    .replace(/(<\/h[345]>)<br>/g, "$1");
}

function answerFootnote(message) {
  const files = [...new Set((message.citations || []).map((item) => item.filename).filter(Boolean))];
  if (files.length) return `参考文件：${files.join("、")}`;
  if (message.answerMode === "database") return "来自热点问答库";
  if (message.answerMode === "rag") return "本次未检索到知识库文件";
  if (message.answerMode === "general_llm") return "";
  return "这条回答对你有帮助吗？";
}

function Message({ message, onFeedback }) {
  const fromAssistant = message.role === "assistant";
  const canRate = fromAssistant && !message.pending && message.requestId;
  return (
    <div className={`msg ${fromAssistant ? "msg-bot" : "msg-me"}`}>
      <div className="msg-meta">
        <span>{fromAssistant ? "智学答" : "我"}</span>
        {message.time ? <time>{message.time}</time> : null}
      </div>
      <div className="msg-body">
        {message.pending
          ? <span className="typing" aria-label="正在生成"><b /><b /><b /> 正在检索资料并组织答案</span>
          : <div className="rich-text" dangerouslySetInnerHTML={{ __html: answerToHtml(message.text) }} />}
      </div>
      {canRate ? (
        <div className="msg-actions">
          <span>{answerFootnote(message)}</span>
          <button type="button" title="有帮助" onClick={() => onFeedback(message.requestId, 1)}><Icon name="thumbUp" size={14} /></button>
          <button type="button" title="没帮助" onClick={() => onFeedback(message.requestId, -1)}><Icon name="thumbDown" size={14} /></button>
          {message.feedback ? <em>{message.feedback}</em> : null}
        </div>
      ) : null}
    </div>
  );
}

function Welcome({ onPick }) {
  return (
    <div className="welcome">
      <div className="welcome-seal">知</div>
      <h1>今天想弄懂什么？</h1>
      <p>问题会先在已接入的学习资料里检索依据，再组织成答案。</p>
      <ul className="starter-list">
        {STARTER_QUESTIONS.map((question) => (
          <li key={question}>
            <button type="button" onClick={() => onPick(question)}>{question}</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function ChatPage({ messages, busy, sources, source, onSourceChange, onSend, onFeedback }) {
  const [draft, setDraft] = useState("");
  const inputRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages]);

  const submit = () => {
    const question = draft.trim();
    if (!question || busy) return;
    setDraft("");
    onSend(question);
  };

  return (
    <div className="chat">
      <div className="thread">
        {messages.length
          ? messages.map((message, index) => <Message key={message.id || `${message.role}-${index}`} message={message} onFeedback={onFeedback} />)
          : <Welcome onPick={(question) => { setDraft(question); inputRef.current?.focus(); }} />}
        <div ref={bottomRef} />
      </div>

      <form className="composer" onSubmit={(event) => { event.preventDefault(); submit(); }}>
        <textarea
          ref={inputRef}
          value={draft}
          rows={2}
          disabled={busy}
          placeholder="输入问题，Enter 发送，Shift + Enter 换行"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(); }
          }}
        />
        <div className="composer-bar">
          <label className="source-picker">
            资料范围
            <select value={source} onChange={(event) => onSourceChange(event.target.value)}>
              <option value="">全部资料</option>
              {sources.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <button className="send" type="submit" title="发送" disabled={busy || !draft.trim()}>
            {busy ? <span className="spinner" /> : <Icon name="arrowUp" size={18} />}
          </button>
        </div>
      </form>
      <p className="fineprint">回答可能有误，重要信息请对照参考文件核实。</p>
    </div>
  );
}
