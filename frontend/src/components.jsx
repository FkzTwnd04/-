/**
 * 各页面共用的小组件：线性图标、卡片、页面标题、状态标签和空状态提示。
 */

const ICON_PATHS = {
  chat: ["M4 5h16v11H9l-5 4z", "M8 9.5h8M8 12.5h5"],
  library: ["M5 4h4v16H5z", "M10 4h4v16h-4z", "m15.5 5 3.8-1 2.7 15.5-3.8 1z"],
  gauge: ["M4 16a8 8 0 1 1 16 0", "m12 16 4-5", "M4 20h16"],
  pulse: ["M3 12h4l2.5-6 5 12 2.5-6h4"],
  plus: ["M12 5v14M5 12h14"],
  arrowUp: ["M12 19V5", "m6 11 6-6 6 6"],
  upload: ["M12 15V4", "m7.5 8.5 4.5-4.5 4.5 4.5", "M4 15v5h16v-5"],
  document: ["M7 3h7l5 5v13H7z", "M14 3v5h5", "M10 13h6M10 17h6"],
  refresh: ["M19 5v5h-5", "M5 19v-5h5", "M18.4 10A7 7 0 0 0 6 7.5M5.6 14A7 7 0 0 0 18 16.5"],
  x: ["m6 6 12 12M18 6 6 18"],
  menu: ["M4 7h16M4 12h16M4 17h10"],
  moon: ["M19 14.5A7.5 7.5 0 0 1 9.5 5a7.5 7.5 0 1 0 9.5 9.5z"],
  sun: ["M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z", "M12 2v2M12 20v2M4 12H2M22 12h-2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5"],
  thumbUp: ["M7 11v9H4v-9z", "M7 11l4-8a2 2 0 0 1 2 2v4h5.5a2 2 0 0 1 2 2.3l-1.2 6.5A2 2 0 0 1 17.3 20H7"],
  thumbDown: ["M7 13V4H4v9z", "M7 13l4 8a2 2 0 0 0 2-2v-4h5.5a2 2 0 0 0 2-2.3l-1.2-6.5A2 2 0 0 0 17.3 4H7"],
};

export function Icon({ name, size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {(ICON_PATHS[name] || ICON_PATHS.chat).map((d) => <path key={d} d={d} />)}
    </svg>
  );
}

export function Panel({ title, description, action, children, className = "" }) {
  return (
    <section className={`panel ${className}`}>
      {title ? (
        <header className="panel-head">
          <div>
            <h2>{title}</h2>
            {description ? <p>{description}</p> : null}
          </div>
          {action}
        </header>
      ) : null}
      {children}
    </section>
  );
}

export function PageHeader({ kicker, title, description, action }) {
  return (
    <header className="page-header">
      <div>
        <span className="kicker">{kicker}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </header>
  );
}

const GOOD_STATES = ["completed", "ingested", "approved"];
const BAD_STATES = ["failed"];

export function StatusTag({ value }) {
  const tone = GOOD_STATES.includes(value) ? "good" : BAD_STATES.includes(value) ? "bad" : "neutral";
  return <span className={`tag tag-${tone}`}>{value || "—"}</span>;
}

export function Placeholder({ children }) {
  return <div className="placeholder">{children}</div>;
}

export function ErrorLine({ children }) {
  return children ? <div className="error-line" role="alert">{children}</div> : null;
}
