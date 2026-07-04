import { useCallback, useEffect, useState, type ReactNode } from "react";
import { postWindwaveOfflineReport, type WindWaveSeriesPoint } from "../adapters/windwaveReportAdapter";
import { type OceanMode, useOceanSession } from "./offlineSession";

type Props = {
  mode: OceanMode;
  ncPath: string | null;
  /** 有路径后自动拉取规则报告 */
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

function scalePoints(values: number[], width: number, height: number, pad = 14): string {
  if (values.length < 1) return "";
  const finite = values.filter((v) => Number.isFinite(v));
  const minV = Math.min(...finite);
  const maxV = Math.max(...finite);
  const span = Math.max(maxV - minV, 1e-6);
  return values
    .map((v, i) => {
      const x = pad + (i / Math.max(values.length - 1, 1)) * (width - pad * 2);
      const y = height - pad - ((v - minV) / span) * (height - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function WindWaveMiniChart({ series }: { series?: WindWaveSeriesPoint[] }) {
  const data = (series ?? []).slice(0, 48);
  if (data.length < 2) {
    return <p className="ocean-dashboard__windwave-chart-empty">暂无风浪时序曲线，请先上传可解析的风浪 NetCDF。</p>;
  }
  const width = 520;
  const height = 154;
  const windObs = data.map((d) => Number(d.wind_observed));
  const windPred = data.map((d) => Number(d.wind_predicted));
  const waveObs = data.map((d) => Number(d.wave_observed));
  const wavePred = data.map((d) => Number(d.wave_predicted));
  return (
    <div className="ocean-dashboard__windwave-chart">
      <div className="ocean-dashboard__windwave-chart-head">
        <strong>风速 / 浪高时序曲线</strong>
        <span>观测值与平滑预测值对照</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="风浪时序曲线">
        <rect x="0" y="0" width={width} height={height} rx="10" />
        <polyline points={scalePoints(windObs, width, height)} className="wind-obs" />
        <polyline points={scalePoints(windPred, width, height)} className="wind-pred" />
        <polyline points={scalePoints(waveObs, width, height)} className="wave-obs" />
        <polyline points={scalePoints(wavePred, width, height)} className="wave-pred" />
      </svg>
      <div className="ocean-dashboard__windwave-legend">
        <span className="wind-obs">风速观测</span>
        <span className="wind-pred">风速预测</span>
        <span className="wave-obs">浪高观测</span>
        <span className="wave-pred">浪高预测</span>
      </div>
    </div>
  );
}

/** 风浪：离线自动 run_detect + 规则报告；异常等级单独一行在报告文本之上。 */
export function WindWavePanel({ mode, ncPath, autoRun = false }: Props) {
  const { setWindwave } = useOceanSession(mode);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [level, setLevel] = useState<string | undefined>(undefined);
  const [index, setIndex] = useState<number | string | undefined>(undefined);
  const [note, setNote] = useState<string | null>(null);
  const [series, setSeries] = useState<WindWaveSeriesPoint[]>([]);

  const load = useCallback(async (path: string) => {
    setErr(null);
    setBusy(true);
    setReport(null);
    setNote(null);
    try {
      const out = await postWindwaveOfflineReport(path);
      const tn = typeof out.typhoon_link_note === "string" ? out.typhoon_link_note : null;
      setReport(out.report_text);
      setLevel(out.anomaly_level);
      setIndex(out.anomaly_index);
      setNote(tn);
      setSeries(Array.isArray(out.wind_wave_series) ? out.wind_wave_series : []);
      setWindwave({
        reportText: out.report_text,
        anomalyLevel: out.anomaly_level,
        anomalyIndex: out.anomaly_index,
        windWaveSeries: Array.isArray(out.wind_wave_series) ? out.wind_wave_series : [],
        typhoonNote: tn,
        typhoonCandidates: Array.isArray(out.typhoon_candidates) ? out.typhoon_candidates : [],
        typhoonEventsPath: out.typhoon_events_path ?? null,
        typhoonQuery: out.typhoon_query ?? null,
        typhoonRetrieval: out.typhoon_retrieval ?? null,
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setReport(null);
      setLevel(undefined);
      setIndex(undefined);
      setSeries([]);
      setWindwave({ reportText: null });
    } finally {
      setBusy(false);
    }
  }, [setWindwave]);

  useEffect(() => {
    if (!autoRun || !ncPath?.trim()) {
      setReport(null);
      setErr(null);
      setLevel(undefined);
      setIndex(undefined);
      setNote(null);
      setSeries([]);
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
            <WindWaveMiniChart series={series} />
            <pre className="ocean-dashboard__windwave-lines">{report}</pre>
          </>
        )}
      </div>
    </div>
  );
}
