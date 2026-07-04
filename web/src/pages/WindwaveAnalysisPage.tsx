import { useMemo, useState, type CSSProperties } from "react";
import { useSearchParams } from "react-router-dom";
import { pollJobUntilDone, postCreateJob, type JobRecord } from "../adapters/jobAdapter";
import { downloadMarkdown, postSaveReport } from "../adapters/reportAdapter";
import {
  levelLabel,
  postStructuredReport,
  type TyphoonCandidate,
  type WindwaveForecastResponse,
} from "../adapters/windwaveForecastAdapter";
import { JobProgressBar } from "../dashboard/JobProgressBar";
import { TaskConfigPanel } from "../dashboard/TaskConfigPanel";
import {
  parseTyphoonDtwMeta,
  type TyphoonQueryMeta,
  TyphoonDtwMetaBar,
  windTrackValues,
  WindTrackSparkline,
} from "../dashboard/typhoonDtwUi";
import { WindWaveForecastChart } from "../dashboard/WindWaveForecastChart";
import { type OceanMode, useOceanSession } from "../dashboard/offlineSession";

/** Phase 3：风浪分析页左-中-右（LSTM 曲线 + 异常等级 + DTW Top-K）。 */
export function WindwaveAnalysisPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const mode: OceanMode = searchParams.get("source") === "realtime" ? "realtime" : "offline";
  const session = useOceanSession(mode);
  const [busy, setBusy] = useState(false);
  const [exportBusy, setExportBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [forecast, setForecast] = useState<WindwaveForecastResponse | null>(null);
  const [reportMd, setReportMd] = useState<string | null>(null);
  const [job, setJob] = useState<JobRecord | null>(null);

  const setSource = (m: OceanMode) => setSearchParams(m === "realtime" ? { source: "realtime" } : {});
  const canRun = !!session.ncPath && session.pipelineArmed;

  const summaryText = useMemo(() => {
    if (!forecast) return "未运行";
    const lvl = forecast.anomaly_level ?? "—";
    const idx = forecast.anomaly_index;
    const n = forecast.series?.length ?? 0;
    return `等级 ${levelLabel(lvl)}${typeof idx === "number" ? ` · 指数 ${idx.toFixed(2)}` : ""} · ${n} 时次`;
  }, [forecast]);

  const onForecast = async () => {
    if (!session.ncPath) {
      setErr("请先在左侧上传并选择 NC");
      return;
    }
    setBusy(true);
    setErr(null);
    setReportMd(null);
    setJob(null);
    try {
      const { job_id } = await postCreateJob({
        type: "windwave_forecast",
        nc_path: session.ncPath,
        top_k: 5,
      });
      const done = await pollJobUntilDone(job_id, setJob);
      const out = (done.result ?? {}) as WindwaveForecastResponse;
      setForecast(out);
      session.setWindwave({
        reportText: null,
        anomalyLevel: out.anomaly_level,
        anomalyIndex: out.anomaly_index,
        windWaveSeries: out.series.map((s) => ({
          step: s.step,
          wind_observed: s.wind_observed,
          wind_predicted: s.wind_predicted,
          wave_observed: s.wave_observed,
          wave_predicted: s.wave_predicted,
        })),
        typhoonNote: out.typhoon_link_note ?? null,
        typhoonCandidates: out.typhoon_candidates ?? [],
        typhoonQuery: (out.typhoon_query as Record<string, unknown>) ?? null,
        typhoonRetrieval: (out.typhoon_retrieval as Record<string, unknown>) ?? null,
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setForecast(null);
    } finally {
      setBusy(false);
      setJob(null);
    }
  };

  const onExportReport = async () => {
    if (!session.ncPath) return;
    setExportBusy(true);
    setErr(null);
    try {
      const out = await postStructuredReport(session.ncPath, 5);
      setReportMd(out.markdown);
      session.setWindwave({ reportText: out.markdown });
      await postSaveReport({
        nc_path: session.ncPath,
        markdown: out.markdown,
        fields: out.fields,
        source: "windwave",
        mode,
        title: `风浪异常报告 · ${session.ncPath.split("/").pop() ?? "NC"}`,
      });
      downloadMarkdown(out.download_name ?? "windwave_report.md", out.markdown);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setExportBusy(false);
    }
  };

  const candidates = forecast?.typhoon_candidates ?? [];
  const dtwMeta = parseTyphoonDtwMeta(forecast?.typhoon_retrieval);

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
        <h2 style={{ margin: 0 }}>风浪预测分析</h2>
        <button type="button" onClick={() => setSource("offline")} style={tabStyle(mode === "offline")}>
          离线
        </button>
        <button type="button" onClick={() => setSource("realtime")} style={tabStyle(mode === "realtime")}>
          准实时
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr 320px", gap: 12, alignItems: "start" }}>
        <div>
          <TaskConfigPanel mode={mode} subsetTask="windwave" />
          <div style={cardStyle}>
            <strong style={{ fontSize: 14 }}>分析任务</strong>
            <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button type="button" disabled={!canRun || busy} onClick={() => void onForecast()}>
                {busy ? "推理中…" : "运行风浪预测"}
              </button>
              <button type="button" disabled={!canRun || exportBusy || !forecast} onClick={() => void onExportReport()}>
                {exportBusy ? "导出中…" : "导出结构化报告"}
              </button>
            </div>
            <p style={{ margin: "8px 0 0", fontSize: 12, color: "#64748b" }}>
              双头 LSTM 预测 + 3σ 异常分级；橙/红预警自动触发 DTW Top-K（异步 job）。
              {!session.pipelineArmed && session.ncPath ? "（已勾选裁剪时请先确认裁剪）" : ""}
            </p>
            <JobProgressBar job={job} label="风浪预测" />
          </div>
        </div>

        <div style={cardStyle}>
          <strong style={{ fontSize: 14 }}>中央预测曲线</strong>
          <p style={{ margin: "6px 0 10px", fontSize: 12, color: "#64748b" }}>{summaryText}</p>
          <WindWaveForecastChart series={forecast?.series ?? []} />
          {forecast?.assessment_note && (
            <p style={{ margin: "10px 0 0", fontSize: 12, color: "#64748b" }}>{forecast.assessment_note}</p>
          )}
          {err && <p style={{ color: "#b91c1c", marginTop: 8, fontSize: 12 }}>{err}</p>}
        </div>

        <div style={cardStyle}>
          <strong style={{ fontSize: 14 }}>异常等级与 DTW Top-K</strong>
          <div style={{ margin: "8px 0 12px" }}>
            <LevelPill level={forecast?.anomaly_level} />
          </div>
          {forecast?.typhoon_link_note && (
            <p style={{ fontSize: 12, color: "#64748b", margin: "0 0 10px" }}>{forecast.typhoon_link_note}</p>
          )}
          <TyphoonDtwMetaBar meta={dtwMeta} query={(forecast?.typhoon_query as TyphoonQueryMeta) ?? null} />
          <div style={{ maxHeight: 280, overflow: "auto", border: "1px solid #e2e8f0", borderRadius: 8 }}>
            {candidates.length ? (
              candidates.map((c, i) => <TopKCard key={i} rank={i + 1} candidate={c} />)
            ) : (
              <p style={{ padding: 10, margin: 0, fontSize: 12, color: "#64748b" }}>
                {forecast ? "未检索到相似台风案例" : "运行预测后显示 Top-K"}
              </p>
            )}
          </div>
          {reportMd && (
            <details style={{ marginTop: 12 }}>
              <summary style={{ fontSize: 13, cursor: "pointer" }}>结构化报告预览</summary>
              <pre style={preStyle}>{reportMd}</pre>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}

function LevelPill({ level }: { level?: string }) {
  const l = (level || "unknown").toLowerCase();
  const bg =
    l === "high" ? "#fef2f2" : l === "medium" ? "#fff7ed" : l === "low" ? "#fefce8" : "#f1f5f9";
  const color =
    l === "high" ? "#b91c1c" : l === "medium" ? "#c2410c" : l === "low" ? "#a16207" : "#475569";
  return (
    <span style={{ display: "inline-block", padding: "4px 10px", borderRadius: 8, background: bg, color, fontSize: 13, fontWeight: 600 }}>
      {levelLabel(level)}
    </span>
  );
}

function TopKCard({ rank, candidate }: { rank: number; candidate: TyphoonCandidate }) {
  const id = String(candidate.event_id ?? candidate.id ?? candidate.name ?? "—");
  const dtw = candidate.dtw_distance;
  const track = windTrackValues(candidate);
  const src = candidate.series_source;
  return (
    <div style={{ padding: "8px 10px", borderBottom: "1px solid #f1f5f9", fontSize: 12 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
        <div style={{ fontWeight: 600 }}>
          Top-{rank} · {id}
        </div>
        <WindTrackSparkline values={track} title={`${id} IBTrACS 中心风速 (m/s)`} />
      </div>
      <div style={{ color: "#64748b", marginTop: 4 }}>
        {candidate.start_time ?? "—"} ~ {candidate.end_time ?? "—"}
      </div>
      <div style={{ color: "#64748b", marginTop: 2 }}>
        时空分 {candidate.score ?? "—"}
        {typeof dtw === "number" ? ` · DTW ${dtw.toFixed(3)}` : ""}
        {track.length > 0 ? ` · 风轨迹 ${track.length} 点` : " · 峰值常数降级"}
        {src ? ` · ${src}` : ""}
      </div>
    </div>
  );
}

const cardStyle: CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 10,
  padding: 10,
};

const preStyle: CSSProperties = {
  marginTop: 8,
  maxHeight: 200,
  overflow: "auto",
  fontSize: 11,
  background: "#f8fafc",
  padding: 8,
  borderRadius: 6,
  whiteSpace: "pre-wrap",
};

const tabStyle = (active: boolean): CSSProperties => ({
  padding: "4px 10px",
  borderRadius: 8,
  border: "1px solid #cbd5e1",
  background: active ? "#0369a1" : "#f8fafc",
  color: active ? "#fff" : "#0f172a",
  fontSize: 12,
});
