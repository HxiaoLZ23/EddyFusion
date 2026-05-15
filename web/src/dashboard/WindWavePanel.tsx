import { useCallback, useEffect, useState, type ReactNode } from "react";
import { postWindwaveOfflineReport } from "../adapters/windwaveReportAdapter";

type Props = {
  ncPath: string | null;
  /** 离线：有路径后自动拉取规则报告 */
  autoRun?: boolean;
};

function pillClass(level: string | undefined): string {
  const l = (level || "").toLowerCase();
  if (l === "high") return "ocean-dashboard__pill ocean-dashboard__pill--high";
  if (l === "medium") return "ocean-dashboard__pill ocean-dashboard__pill--mid";
  if (l === "low") return "ocean-dashboard__pill ocean-dashboard__pill--low";
  return "ocean-dashboard__pill ocean-dashboard__pill--unknown";
}

function LevelRow({ children }: { children: ReactNode }) {
  return <div className="ocean-dashboard__windwave-level-row">{children}</div>;
}

/** 风浪：离线自动 run_detect + 规则报告；异常等级单独一行在报告文本之上。 */
export function WindWavePanel({ ncPath, autoRun = false }: Props) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [level, setLevel] = useState<string | undefined>(undefined);
  const [index, setIndex] = useState<number | string | undefined>(undefined);
  const [note, setNote] = useState<string | null>(null);

  const load = useCallback(async (path: string) => {
    setErr(null);
    setBusy(true);
    setReport(null);
    setNote(null);
    try {
      const out = await postWindwaveOfflineReport(path);
      setReport(out.report_text);
      setLevel(out.anomaly_level);
      setIndex(out.anomaly_index);
      setNote(typeof out.typhoon_link_note === "string" ? out.typhoon_link_note : null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setReport(null);
      setLevel(undefined);
      setIndex(undefined);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    if (!autoRun || !ncPath?.trim()) {
      setReport(null);
      setErr(null);
      setLevel(undefined);
      setIndex(undefined);
      setNote(null);
      return;
    }
    void load(ncPath);
  }, [autoRun, ncPath, load]);

  if (!autoRun) {
    const hasNc = !!ncPath;
    return (
      <div className="ocean-dashboard__panel ocean-dashboard__panel--windwave">
        <h3 className="ocean-dashboard__panel-head">风浪区块</h3>
        <LevelRow>
          <div className="ocean-dashboard__pill ocean-dashboard__pill--unknown">
            {hasNc ? "异常等级: 实时模式未接自动报告" : "异常等级: 无数据"}
          </div>
        </LevelRow>
        <div className="ocean-dashboard__panel-body ocean-dashboard__panel-body--windwave">
          <pre className="ocean-dashboard__windwave-lines">
            {`异常等级: ${hasNc ? "-" : "无数据"}
异常指数: -
说明: 实时同屏下风浪报告未自动触发；请使用离线单文件上传以生成规则报告。`}
          </pre>
        </div>
      </div>
    );
  }

  if (!ncPath?.trim()) {
    return (
      <div className="ocean-dashboard__panel ocean-dashboard__panel--windwave">
        <h3 className="ocean-dashboard__panel-head">风浪区块 · 规则报告</h3>
        <LevelRow>
          <div className="ocean-dashboard__pill ocean-dashboard__pill--unknown">等待上传 NC</div>
        </LevelRow>
        <div className="ocean-dashboard__panel-body ocean-dashboard__panel-body--windwave" />
      </div>
    );
  }

  return (
    <div className="ocean-dashboard__panel ocean-dashboard__panel--windwave">
      <h3 className="ocean-dashboard__panel-head">风浪区块 · 规则报告</h3>
      <LevelRow>
        {busy ? (
          <span className="ocean-dashboard__windwave-level-muted">正在解析风浪并判定等级…</span>
        ) : err ? (
          <div className="ocean-dashboard__pill ocean-dashboard__pill--unknown">异常等级: 未能生成（见下方说明）</div>
        ) : report ? (
          <div className={pillClass(level)}>
            异常等级: {level ?? "未知"}
            {index !== undefined && index !== null ? ` · 指数 ${String(index)}` : ""}
          </div>
        ) : (
          <span className="ocean-dashboard__windwave-level-muted">—</span>
        )}
      </LevelRow>
      <div className="ocean-dashboard__panel-body ocean-dashboard__panel-body--windwave">
        {busy && <p className="ocean-dashboard__windwave-body-lead">正在从 NC 提取风浪并生成报告…</p>}
        {err && (
          <p className="ocean-dashboard__windwave-err">
            {err}
            <br />
            <small>常见原因：NC 不含 u10/v10 或有效波高等可解析变量（与 Streamlit 风浪页一致）。</small>
          </p>
        )}
        {!busy && !err && report && (
          <>
            {note && <p className="ocean-dashboard__windwave-note">{note}</p>}
            <pre className="ocean-dashboard__windwave-lines">{report}</pre>
          </>
        )}
      </div>
    </div>
  );
}
