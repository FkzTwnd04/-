/**
 * 评测页：上传人工标注的 JSONL 评测集，运行 RAG 或意图识别评测，并可下载最近一次报告。
 */
import { useState } from "react";
import { pollUntilDone, requestJson } from "../api.js";
import { ErrorLine, Icon, PageHeader, Panel, StatusTag } from "../components.jsx";

const SAMPLE_LINES = {
  rag: '{"id":"样本编号","query":"问题","reference_answer":"人工核对答案","relevant_chunk_ids":["真实 Chunk ID"]}',
  intent: '{"id":"样本编号","query":"问题","expected_intent":"人工确认意图"}',
};

export default function EvaluationPage({ notify }) {
  const [mode, setMode] = useState("rag");
  const [runName, setRunName] = useState("智学答评测");
  const [dataset, setDataset] = useState(null);
  const [job, setJob] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  async function start(event) {
    event.preventDefault();
    if (!dataset) { setError("请先选择 JSONL 评测集。"); return; }
    setRunning(true);
    setError("");
    setJob(null);
    const form = new FormData();
    form.append("file", dataset);
    form.append("mode", mode);
    form.append("run_name", runName);
    try {
      const created = await requestJson("/api/eval/runs", { method: "POST", body: form });
      setJob(created);
      await pollUntilDone(`/api/eval/runs/${created.job_id}`, (state) => {
        setJob(state);
        return ["completed", "failed"].includes(state.status);
      });
    } catch (issue) {
      setError(issue.message);
    } finally {
      setRunning(false);
    }
  }

  async function downloadLatest() {
    try {
      const report = await requestJson("/api/eval/latest");
      const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = "zhixueda-latest-evaluation.json";
      link.click();
      URL.revokeObjectURL(url);
    } catch (issue) {
      notify(issue.message);
    }
  }

  return (
    <div className="page">
      <PageHeader
        kicker="质量验证"
        title="离线评测"
        description="用人工标注的数据检验检索命中和回答质量。"
        action={<button type="button" className="btn" onClick={() => void downloadLatest()}>下载最近报告</button>}
      />

      <Panel>
        <form className="eval-form" onSubmit={start}>
          <div className="field-grid">
            <label className="field">
              <span>评测类型</span>
              <select value={mode} onChange={(event) => setMode(event.target.value)}>
                <option value="rag">RAG 问答与检索</option>
                <option value="intent">意图识别</option>
              </select>
            </label>
            <label className="field">
              <span>报告名称</span>
              <input value={runName} maxLength={60} onChange={(event) => setRunName(event.target.value)} />
            </label>
          </div>
          <label className="dropzone">
            <Icon name="document" />
            <span>{dataset ? dataset.name : "点击选择 JSONL 评测集"}</span>
            <input type="file" accept=".jsonl,application/jsonl" onChange={(event) => setDataset(event.target.files?.[0] || null)} />
          </label>
          <button type="submit" className="btn btn-primary" disabled={running}>{running ? "评测运行中…" : "开始评测"}</button>
        </form>
        <div className="hint-box">
          <strong>每行一个 JSON 对象，例如：</strong>
          <code>{SAMPLE_LINES[mode]}</code>
          <p>检索评测里的 Chunk ID 必须来自当前知识库；不填写时跳过检索指标。</p>
        </div>
      </Panel>

      <ErrorLine>{error}</ErrorLine>
      {job ? (
        <Panel title={job.run_name || job.job_id} action={<StatusTag value={job.status} />}>
          {job.error ? <p className="text-bad">{job.error}</p> : null}
          {job.status === "completed" ? <pre className="report">{JSON.stringify(job.result || job, null, 2)}</pre> : null}
        </Panel>
      ) : null}
    </div>
  );
}
