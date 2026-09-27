/**
 * 运营页：按答案来源层统计近 7 天请求，审核高频问题候选，并查看脱敏后的请求日志。
 */
import { useCallback, useEffect, useState } from "react";
import { formatTime, postJson, requestJson } from "../api.js";
import { ErrorLine, Icon, PageHeader, Panel, Placeholder, StatusTag } from "../components.jsx";

const percent = (value) => `${Math.round((value || 0) * 100)}%`;

export default function OperationsPage({ notify }) {
  const [layers, setLayers] = useState({});
  const [requests, setRequests] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const reload = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [stats, logs, pending] = await Promise.all([
        requestJson("/api/ops/stats?days=7"),
        requestJson("/api/ops/requests?limit=50"),
        requestJson("/api/ops/candidates?status=pending&limit=100"),
      ]);
      setLayers(stats.layers || {});
      setRequests(logs.items || []);
      setCandidates(pending.items || []);
    } catch (issue) {
      setError(`${issue.message}。请确认后端服务和 MySQL 已启动。`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void reload(); }, [reload]);

  async function buildCandidates() {
    try {
      const result = await requestJson("/api/ops/candidates/build?min_freq=3&days=7", { method: "POST" });
      notify(`发现并登记 ${result.created_or_updated} 个候选问题。`);
      await reload();
    } catch (issue) {
      setError(issue.message);
    }
  }

  async function review(candidate, approved) {
    const answer = approved ? window.prompt(`为“${candidate.cluster_question}”填写人工确认的答案：`) : "";
    if (approved && !answer?.trim()) return;
    try {
      await postJson(`/api/ops/candidates/${candidate.id}/review`, { approved, answer: answer || "" });
      await reload();
    } catch (issue) {
      setError(issue.message);
    }
  }

  const layerEntries = Object.entries(layers);

  return (
    <div className="page">
      <PageHeader
        kicker="系统观测"
        title="运营与请求日志"
        description="各答案来源层的请求量、耗时和点踩率，以及需要人工补充答案的高频问题。"
        action={<button type="button" className="btn" onClick={() => void reload()}><Icon name="refresh" size={15} />刷新</button>}
      />
      <ErrorLine>{error}</ErrorLine>

      <div className="stat-row">
        {layerEntries.length ? layerEntries.map(([layer, item]) => (
          <div className="stat" key={layer}>
            <span>{layer}</span>
            <strong>{item.requests}</strong>
            <small>占比 {percent(item.share)} · 平均 {Math.round(item.average_latency_ms)} ms · 点踩 {percent(item.thumbs_down_rate)}</small>
          </div>
        )) : (
          <div className="stat">
            <span>最近 7 天</span>
            <strong>—</strong>
            <small>{loading ? "正在读取统计" : "暂无请求记录"}</small>
          </div>
        )}
      </div>

      <Panel
        title="待审核问题"
        description="由高频的完整 RAG 问题聚合而来，只有人工填写答案后才会发布。"
        action={<button type="button" className="btn btn-primary" onClick={() => void buildCandidates()}>生成候选</button>}
      >
        {candidates.length ? (
          <ul className="review-list">
            {candidates.map((candidate) => (
              <li key={candidate.id}>
                <div>
                  <strong>{candidate.cluster_question}</strong>
                  <small>出现 {candidate.frequency} 次 · {candidate.sample_queries?.join(" / ")}</small>
                </div>
                <div className="review-actions">
                  <button type="button" className="btn" onClick={() => void review(candidate, false)}>忽略</button>
                  <button type="button" className="btn btn-primary" onClick={() => void review(candidate, true)}>补充答案</button>
                </div>
              </li>
            ))}
          </ul>
        ) : <Placeholder>{loading ? "正在读取候选…" : "当前没有待审核问题。"}</Placeholder>}
      </Panel>

      <Panel
        title="最近请求"
        description="问题和回答写入日志前已对手机号、邮箱和身份证号脱敏。"
        action={<span className="muted">{requests.length} 条</span>}
      >
        {requests.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>问题 / 回答摘要</th><th>命中层</th><th>意图</th><th>耗时</th><th>评价</th><th>时间</th></tr></thead>
              <tbody>
                {requests.map((row) => (
                  <tr key={row.request_id}>
                    <td><strong>{row.query}</strong><small>{row.answer}</small></td>
                    <td><StatusTag value={row.hit_layer} /></td>
                    <td>{row.intent || "—"}</td>
                    <td>{row.latency_ms} ms</td>
                    <td>{row.rating === 1 ? "有帮助" : row.rating === -1 ? "没帮助" : "—"}</td>
                    <td>{formatTime(row.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <Placeholder>{loading ? "正在读取日志…" : "暂无请求日志。"}</Placeholder>}
      </Panel>
    </div>
  );
}
