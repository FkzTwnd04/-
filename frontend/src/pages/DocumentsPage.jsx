/**
 * 知识库页：上传学习资料并跟踪入库任务，列出文档台账中的处理状态，可一键重试失败任务。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { formatTime, pollUntilDone, requestJson } from "../api.js";
import { ErrorLine, Icon, PageHeader, Panel, Placeholder, StatusTag } from "../components.jsx";

const FINISHED = ["completed", "failed", "ingested", "duplicate"];

export default function DocumentsPage({ sources, notify }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [target, setTarget] = useState(sources[0] || "ai");
  const [upload, setUpload] = useState(null);
  const picker = useRef(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await requestJson("/api/docs?limit=100");
      setDocuments(data.documents || []);
    } catch (issue) {
      setError(issue.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void reload(); }, [reload]);

  async function send(file) {
    if (!file) return;
    setError("");
    setUpload({ name: file.name, status: "上传中" });
    const form = new FormData();
    form.append("file", file);
    form.append("source", target || "ai");
    try {
      const task = await requestJson("/api/docs/upload", { method: "POST", body: form });
      setUpload({ name: file.name, status: "正在入库" });
      await pollUntilDone(`/api/docs/tasks/${task.task_id}`, (state) => {
        setUpload({ name: file.name, status: state.status, error: state.error });
        return FINISHED.includes(state.status);
      });
      await reload();
    } catch (issue) {
      setUpload({ name: file.name, status: "失败", error: issue.message });
      setError(issue.message);
    }
  }

  async function retryFailed() {
    try {
      const result = await requestJson("/api/docs/retry", { method: "POST" });
      notify(`已重新提交 ${result.submitted || 0} 个失败任务。`);
      await reload();
    } catch (issue) {
      setError(issue.message);
    }
  }

  return (
    <div className="page">
      <PageHeader
        kicker="知识管理"
        title="文档知识库"
        description="上传学习资料后，系统会解析内容、切分文本并写入向量库。"
        action={<button type="button" className="btn" onClick={() => void reload()}><Icon name="refresh" size={15} />刷新</button>}
      />

      <Panel title="导入文档" description="支持 PDF、Word(.docx)、PPT(.pptx)、图片、Markdown 和纯文本。">
        <div className="upload-row">
          <label className="field">
            <span>归入学科</span>
            <select value={target} onChange={(event) => setTarget(event.target.value)}>
              {(sources.length ? sources : ["ai"]).map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <input ref={picker} type="file" hidden onChange={(event) => { void send(event.target.files?.[0]); event.target.value = ""; }} />
          <button type="button" className="btn btn-primary" onClick={() => picker.current?.click()}><Icon name="upload" size={15} />选择文件</button>
        </div>
        {upload ? (
          <div className="upload-state">
            <Icon name="document" size={16} />
            <span>{upload.name}</span>
            <StatusTag value={upload.status} />
            {upload.error ? <small className="text-bad">{upload.error}</small> : null}
          </div>
        ) : null}
      </Panel>

      <Panel
        title="文档台账"
        description="每个文件的入库状态、文本块数量和重试次数。"
        action={<button type="button" className="btn" onClick={() => void retryFailed()}>重试失败任务</button>}
      >
        <ErrorLine>{error}</ErrorLine>
        {loading ? <Placeholder>正在读取文档状态…</Placeholder> : documents.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>文档</th><th>状态</th><th>文本块</th><th>重试</th><th>更新时间</th></tr></thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.doc_id}>
                    <td><strong>{doc.title || doc.source_path}</strong><small>{doc.source_path}</small></td>
                    <td><StatusTag value={doc.status} />{doc.error_msg ? <small className="text-bad">{doc.error_msg}</small> : null}</td>
                    <td>{doc.chunk_count || 0}</td>
                    <td>{doc.retry_count || 0}</td>
                    <td>{formatTime(doc.updated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <Placeholder>还没有文档，上传一个资料文件后会出现在这里。</Placeholder>}
      </Panel>
    </div>
  );
}
