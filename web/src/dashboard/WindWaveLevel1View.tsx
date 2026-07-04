import { useState } from "react";
import { Link } from "react-router-dom";
import { postWindwaveLlmReport } from "../adapters/windwaveLlmAdapter";
import type { TyphoonCandidateRow } from "./offlineSession";
import { type OceanMode, useOceanSession } from "./offlineSession";
import {
  parseTyphoonDtwMeta,
  type TyphoonQueryMeta,
  TyphoonDtwMetaBar,
  windTrackValues,
  WindTrackSparkline,
} from "./typhoonDtwUi";
import { WindWaveMiniChart } from "./WindWavePanel";

type Props = {
  mode: OceanMode;
  reportText: string | null;
  anomalyLevel?: string;
  typhoonNote?: string | null;
};

function levelPillClass(level: string | undefined): string {
  const l = (level || "").toLowerCase();
  if (l === "high") return "ocean-dashboard__pill ocean-dashboard__pill--high";
  if (l === "medium") return "ocean-dashboard__pill ocean-dashboard__pill--mid";
  if (l === "low") return "ocean-dashboard__pill ocean-dashboard__pill--low";
  return "ocean-dashboard__pill ocean-dashboard__pill--unknown";
}

function candidateLabel(c: TyphoonCandidateRow): string {
  return String(c.event_id ?? c.id ?? c.name ?? "—");
}

/** 规划 §6.4：风浪 L1 — 左报告 + 右栏 LLM（主）+ 台风查询 */
export function WindWaveLevel1View({ mode, reportText, anomalyLevel, typhoonNote }: Props) {
  const { ncPath, windwaveLlm, setWindwaveLlm, windwave } = useOceanSession(mode);
  const [llmBusy, setLlmBusy] = useState(false);

  const candidates = windwave.typhoonCandidates ?? [];
  const dtwMeta = parseTyphoonDtwMeta(windwave.typhoonRetrieval);
  const kbPath = mode === "offline" ? "/offline/typhoon-kb" : "/realtime/typhoon-kb";
  const kbPrefill = windwave.typhoonQuery
    ? { prefill: { ...windwave.typhoonQuery, events_json_path: windwave.typhoonEventsPath ?? undefined } }
    : undefined;

  const onLlm = async () => {
    if (!ncPath?.trim()) return;
    setLlmBusy(true);
    setWindwaveLlm({ error: null });
    try {
      const r = await postWindwaveLlmReport(ncPath);
      setWindwaveLlm({
        summaryAnomaly: r.summary_anomaly,
        impact: r.impact,
        historicalAnalogy: r.historical_analogy,
        actions: r.actions,
        error: null,
      });
    } catch (e) {
      setWindwaveLlm({ error: e instanceof Error ? e.message : String(e) });
    } finally {
      setLlmBusy(false);
    }
  };

  return (
    <div className="ocean-dashboard__l1-windwave">
      <div className="ocean-dashboard__l1-windwave-head">
        <h4 className="ocean-dashboard__l1-windwave-title">风浪 · 结构化报告与扩展</h4>
        {anomalyLevel && <span className={levelPillClass(anomalyLevel)}>异常等级: {anomalyLevel}</span>}
      </div>
      <div className="ocean-dashboard__l1-windwave-body">
        <section className="ocean-dashboard__l1-windwave-report">
          <h5>规则报告（全文）</h5>
          <WindWaveMiniChart series={windwave.windWaveSeries} />
          {reportText ? (
            <pre className="ocean-dashboard__l1-windwave-pre">{reportText}</pre>
          ) : (
            <p className="ocean-dashboard__l1-muted">暂无报告，请先在同屏完成风浪自动分析后再进入本页。</p>
          )}
        </section>
        <aside className="ocean-dashboard__l1-windwave-side">
          <section className="ocean-dashboard__l1-windwave-card ocean-dashboard__l1-windwave-card--llm">
            <div className="ocean-dashboard__l1-windwave-card-head">
              <h5>智能解读（LLM）</h5>
              <button
                type="button"
                className="ocean-dashboard__eddy-btn"
                disabled={llmBusy || !ncPath}
                onClick={() => void onLlm()}
              >
                {llmBusy ? "生成中…" : "生成智能解读"}
              </button>
            </div>
            <p className="ocean-dashboard__l1-muted">
              后端读取 <code>config/dashscope.local.json</code> 或环境变量；勿在前端填密钥。
            </p>
            <div className="ocean-dashboard__llm-scroll">
              {windwaveLlm.error && <p className="ocean-dashboard__hydro-buffer-err">{windwaveLlm.error}</p>}
              {!windwaveLlm.summaryAnomaly && !windwaveLlm.error && (
                <p className="ocean-dashboard__l1-muted">点击上方按钮生成；结果将保留在本会话内。</p>
              )}
              {windwaveLlm.summaryAnomaly && (
                <div className="ocean-dashboard__llm-sections">
                  <p>
                    <strong>综述</strong>
                    <br />
                    {windwaveLlm.summaryAnomaly}
                  </p>
                  {windwaveLlm.impact && (
                    <p>
                      <strong>影响</strong>
                      <br />
                      {windwaveLlm.impact}
                    </p>
                  )}
                  {windwaveLlm.historicalAnalogy && (
                    <p>
                      <strong>历史类比</strong>
                      <br />
                      {windwaveLlm.historicalAnalogy}
                    </p>
                  )}
                  {(windwaveLlm.actions?.length ?? 0) > 0 && (
                    <div>
                      <strong>建议</strong>
                      <ol className="ocean-dashboard__llm-actions">
                        {windwaveLlm.actions!.map((a, i) => (
                          <li key={i}>{a}</li>
                        ))}
                      </ol>
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>

          <section className="ocean-dashboard__l1-windwave-card ocean-dashboard__l1-windwave-card--kb">
            <h5>台风查询 · 相似事件</h5>
            {typhoonNote && <p className="ocean-dashboard__l1-windwave-note">{typhoonNote}</p>}
            <TyphoonDtwMetaBar meta={dtwMeta} query={(windwave.typhoonQuery as TyphoonQueryMeta) ?? null} />
            {candidates.length > 0 ? (
              <div className="ocean-dashboard__eddy-drawer-table-wrap">
                <table className="ocean-dashboard__eddy-drawer-table">
                  <thead>
                    <tr>
                      <th>事件</th>
                      <th>时间窗</th>
                      <th>匹配分</th>
                      <th>DTW</th>
                      <th>风轨迹</th>
                    </tr>
                  </thead>
                  <tbody>
                    {candidates.slice(0, 8).map((c, i) => {
                      const track = windTrackValues(c);
                      return (
                        <tr key={`${candidateLabel(c)}-${i}`}>
                          <td title={candidateLabel(c)}>{candidateLabel(c)}</td>
                          <td className="ocean-dashboard__eddy-drawer-time">
                            {c.start_time && c.end_time ? `${c.start_time} ~ ${c.end_time}` : "—"}
                          </td>
                          <td>{c.score != null ? Number(c.score).toFixed(3) : "—"}</td>
                          <td>{c.dtw_distance != null ? Number(c.dtw_distance).toFixed(3) : "—"}</td>
                          <td>
                            {track.length > 0 ? (
                              <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                                <WindTrackSparkline values={track} />
                                <span style={{ fontSize: 11, color: "#64748b" }}>{track.length} 点</span>
                              </span>
                            ) : (
                              <span style={{ fontSize: 11, color: "#94a3b8" }}>峰值降级</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="ocean-dashboard__l1-muted">
                未检索到时空匹配的候选台风（索引已存在时多为查询窗与事件 bbox 无交集）。
                完整库构建：<code>scripts/run_typhoon_kb.ps1</code>；完成后请在同屏重新生成风浪报告。
                {windwave.typhoonEventsPath && (
                  <>
                    <br />
                    索引：{windwave.typhoonEventsPath}
                  </>
                )}
                {typhoonNote && (
                  <>
                    <br />
                    {typhoonNote}
                  </>
                )}
              </p>
            )}
            <Link className="ocean-dashboard__l1-link" to={kbPath} state={kbPrefill}>
              打开台风查询页 →
            </Link>
          </section>
        </aside>
      </div>
    </div>
  );
}
