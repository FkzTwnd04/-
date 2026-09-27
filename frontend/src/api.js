/**
 * 前端访问本项目 FastAPI 的全部工具函数：JSON 请求、异步任务轮询和 WebSocket 流式问答。
 */

export async function requestJson(path, options = {}) {
  const response = await fetch(path, options);
  const isJson = (response.headers.get("content-type") || "").includes("application/json");
  const body = isJson ? await response.json() : {};
  if (!response.ok) {
    throw new Error(body.detail || body.error || `请求失败（${response.status}）`);
  }
  return body;
}

export function postJson(path, payload) {
  return requestJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** 每秒查询一次任务状态，isDone 返回 true 时结束；最多等待 10 分钟。 */
export async function pollUntilDone(path, isDone, { intervalMs = 1000, maxTries = 600 } = {}) {
  for (let tries = 0; tries < maxTries; tries += 1) {
    const result = await requestJson(path);
    if (isDone(result)) return result;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error("等待任务超时，请稍后刷新状态。");
}

/**
 * 通过 /api/stream 获取流式回答。onToken 收到累计文本，返回的 Promise 以结束帧的数据完成。
 */
export function streamQuestion(payload, onToken) {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/api/stream`);
  let text = "";
  let settled = false;

  return new Promise((resolve, reject) => {
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      socket.close();
      callback(value);
    };
    socket.addEventListener("open", () => socket.send(JSON.stringify(payload)));
    socket.addEventListener("message", (event) => {
      let frame;
      try { frame = JSON.parse(event.data); } catch { return; }
      if (frame.type === "token") {
        text += frame.token || "";
        onToken(text);
      } else if (frame.type === "end") {
        finish(resolve, { ...frame, text });
      } else if (frame.type === "error") {
        finish(reject, new Error(frame.error || "流式回答失败"));
      }
    });
    socket.addEventListener("error", () => finish(reject, new Error("WebSocket 连接失败")));
    socket.addEventListener("close", () => finish(resolve, { text }));
  });
}

export function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

export function clockTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
