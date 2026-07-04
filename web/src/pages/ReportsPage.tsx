import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import { Link } from "react-router-dom";
import {
  downloadMarkdown,
  fetchReportById,
  fetchReportHistory,
  type ReportListItem,
  type SavedReport,
} from "../adapters/reportAdapter";
import { postWindwaveLlmReport } from "../adapters/windwaveLlmAdapter";
import { postStructuredReport } from "../adapters/windwaveForecastAdapter";
import { levelLabel } from "../adapters/windwaveForecastAdapter";
import { useOceanSession, type EddyJobHistoryRow, type OceanMode, type WindwaveLlmSnapshot } from "../dashboard/offlineSession";

/** Phase 4：报告管理 — 历史列表、预览、下载与再导出。 */
export function ReportsPage() {
  const offline = useOceanSession("offline");
  const realtime = useOceanSession("realtime");
  const [tab, setTab] = useState<"saved" | "tasks">("saved");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [reports, setReports] = useState<ReportListItem[]>([]);
  const [selected, setSelected] = useState<SavedReport | null>(null);
  const [previewTab, setPreviewTab] = useState<"structured" | "llm">("structured");
  const [reexportBusy, setReexportBusy] = useState(false);
  const [llmBusy, setLlmBusy] = useState(false);
  const [llmView, setLlmView] = useState<WindwaveLlmSnapshot | null>(null);

  const taskRows = useMemo(() => {
    const merge = (
      mode: OceanMode,
      eddy: EddyJobHistoryRow[],
      ncPath: string | null,
      level?: string,
    ) => {
      const rows: { key: string; mode: OceanMode; kind: string; at: string; label: string; status: string; extra?: string }[] = [];
      for (const j of eddy) {
        rows.push({
          key: `${mode}-eddy-${j.id}`,
          mode,
          kind: "涡旋双路 MP4",
          at: j.at,
          label: j.ncLabel,
          status: j.status,
          extra: j.nFrames != null ? `${j.nFrames} 帧` : undefined,
        });
      }
      if (ncPath && level) {
        rows.push({
          key: `${mode}-wind-${ncPath}`,
          mode,
          kind: "风浪预测",
          at: "会话内",
          label: ncPath,
          status: "success",
          extra: levelLabel(level),
        });
      }
      return rows;
    };
    return [
      ...merge("offline", offline.eddyHistory, offline.ncPath, offline.windwave.anomalyLevel),
      ...merge("realtime", realtime.eddyHistory, realtime.ncPath, realtime.windwave.anomalyLevel),
    ];
  }, [offline, realtime]);

  const loadHistory = useCallback(async (autoSelectFirst = false) => {
    setBusy(true);
    setErr(null);
    try {
      const rows = await fetchReportHistory(50);
      setReports(rows);
      if (autoSelectFirst && rows.length) {
        const full = await fetchReportById(rows[0].id);
        setSelected(full);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setReports([]);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void loadHistory(true);
  }, [loadHistory]);

  const onSelect = async (id: string) => {
    setErr(null);
    setLlmView(null);
    try {
      const full = await fetchReportById(id);
      setSelected(full);
      const savedLlm = full.fields?.llm as WindwaveLlmSnapshot | undefined;
      if (savedLlm?.summaryAnomaly) {
        setLlmView(savedLlm);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  const onGenerateLlm = async () => {
    const nc = selected?.nc_path;
    if (!nc) return;
    setLlmBusy(true);
    setErr(null);
    try {
      const r = await postWindwaveLlmReport(nc);
      const snap: WindwaveLlmSnapshot = {
        summaryAnomaly: r.summary_anomaly,
        impact: r.impact,
        historicalAnalogy: r.historical_analogy,
        actions: r.actions,
        error: null,
      };
      setLlmView(snap);
      offline.setWindwaveLlm(snap);
      realtime.setWindwaveLlm(snap);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setLlmView({ error: msg });
      setErr(msg);
    } finally {
      setLlmBusy(false);
    }
  };

  const onReexport = async () => {
    if (!selected?.nc_path) return;
    setReexportBusy(true);
    setErr(null);
    try {
      const out = await postStructuredReport(selected.nc_path, 5);
      setSelected({ ...selected, markdown: out.markdown, fields: out.fields });
      downloadMarkdown(out.download_name ?? `report_${selected.id}.md`, out.markdown);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setReexportBusy(false);
    }
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
        <h2 style={{ margin: 0 }}>报告管理</h2>
        <button type="button" style={tabBtn(tab === "saved")} onClick={() => setTab("saved")}>
          已保存报告
        </button>
        <button type="button" style={tabBtn(tab === "tasks")} onClick={() => setTab("tasks")}>
          会话任务记录
        </button>
        <button type="button" onClick={() => void loadHistory(false)} disabled={busy} style={{ marginLeft: "auto", fontSize: 12 }}>
          {busy ? "刷新中…" : "刷新列表"}
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: 12, alignItems: "start" }}>
        <div style={cardStyle}>
          {tab === "saved" ? (
            <>
              <strong style={{ fontSize: 14 }}>历史报告（{reports.length}）</strong>
              <p style={{ margin: "6px 0 10px", fontSize: 12, color: "#64748b" }}>
                风浪页「导出结构化报告」后自动归档；右侧可预览规则报告与 LLM 智能解读。
              </p>
              <div style={{ maxHeight: 520, overflow: "auto" }}>
                {reports.map((r) => (
                  <button
                    key={r.id}
                    type="button"
                    onClick={() => void onSelect(r.id)}
                    style={listBtn(selected?.id === r.id)}
                  >
                    <div style={{ fontWeight: 600, fontSize: 13 }}>{r.title ?? r.id}</div>
                    <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
                      {r.created_at_iso ?? "—"} · {levelLabel(r.anomaly_level)} · {r.mode ?? "offline"}
                    </div>
                    <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2, wordBreak: "break-all" }}>
                      {r.nc_path}
                    </div>
                  </button>
                ))}
                {!reports.length && !busy && (
                  <p style={{ fontSize: 12, color: "#64748b" }}>
                    暂无已保存报告。请先在 <Link to="/windwave">风浪分析</Link> 运行预测并导出。
                  </p>
                )}
              </div>
            </>
          ) : (
            <>
              <strong style={{ fontSize: 14 }}>会话任务（{taskRows.length}）</strong>
              <p style={{ margin: "6px 0 10px", fontSize: 12, color: "#64748b" }}>
                当前浏览器会话内的涡旋/风浪任务摘要（未持久化）。
              </p>
              <div style={{ maxHeight: 520, overflow: "auto" }}>
                {taskRows.map((t) => (
                  <div key={t.key} style={{ padding: "8px 0", borderBottom: "1px solid #f1f5f9", fontSize: 12 }}>
                    <div style={{ fontWeight: 600 }}>
                      {t.kind} · {t.mode}
                    </div>
                    <div style={{ color: "#64748b" }}>{t.at} · {t.status}{t.extra ? ` · ${t.extra}` : ""}</div>
                    <div style={{ color: "#94a3b8", wordBreak: "break-all" }}>{t.label}</div>
                  </div>
                ))}
                {!taskRows.length && (
                  <p style={{ fontSize: 12, color: "#64748b" }}>
                    暂无任务记录。请从 <Link to="/monitor">监测总览</Link> 上传 NC 并运行分析。
                  </p>
                )}
              </div>
            </>
          )}
        </div>

        <div style={cardStyle}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
            <strong style={{ fontSize: 14 }}>报告预览</strong>
            <button type="button" style={tabBtn(previewTab === "structured")} onClick={() => setPreviewTab("structured")}>
              结构化报告
            </button>
            <button type="button" style={tabBtn(previewTab === "llm")} onClick={() => setPreviewTab("llm")}>
              LLM 智能解读
            </button>
            {selected && previewTab === "structured" && (
              <>
                <button
                  type="button"
                  disabled={reexportBusy}
                  onClick={() => downloadMarkdown(`${selected.id}.md`, selected.markdown)}
                >
                  下载 Markdown
                </button>
                <button type="button" disabled={reexportBusy || !selected.nc_path} onClick={() => void onReexport()}>
                  {reexportBusy ? "再导出中…" : "再导出（重跑 API）"}
                </button>
              </>
            )}
            {selected && previewTab === "llm" && (
              <button type="button" disabled={llmBusy || !selected.nc_path} onClick={() => void onGenerateLlm()}>
                {llmBusy ? "生成中…" : "生成智能解读"}
              </button>
            )}
            {selected?.nc_path && (
              <Link to="/windwave" style={{ fontSize: 12, marginLeft: "auto" }}>
                打开风浪分析
              </Link>
            )}
          </div>
          {selected ? (
            <>
              <p style={{ margin: "0 0 8px", fontSize: 12, color: "#64748b" }}>
                {selected.title} · {levelLabel(selected.anomaly_level)} · {selected.nc_path}
              </p>
              {previewTab === "structured" ? (
                <pre style={preStyle}>{selected.markdown}</pre>
              ) : (
                <LlmPreview
                  llm={llmView ?? offline.windwaveLlm}
                  busy={llmBusy}
                  onGenerate={() => void onGenerateLlm()}
                  ncReady={!!selected.nc_path}
                />
              )}
            </>
          ) : (
            <p style={{ fontSize: 13, color: "#64748b" }}>从左侧选择一条报告以预览全文。</p>
          )}
          {err && <p style={{ color: "#b91c1c", fontSize: 12, marginTop: 8 }}>{err}</p>}
        </div>
      </div>

      <div style={{ marginTop: 16, display: "flex", gap: 10, flexWrap: "wrap" }}>
        <QuickLink to="/monitor" label="监测总览" desc="上传 NC · 裁剪 · 双路 MP4" />
        <QuickLink to="/eddy" label="涡旋分析" desc="YOLO 单帧预览 · 统计表" />
        <QuickLink to="/windwave" label="风浪分析" desc="LSTM 曲线 · DTW Top-K" />
      </div>
    </div>
  );
}

function LlmPreview({
  llm,
  busy,
  onGenerate,
  ncReady,
}: {
  llm: WindwaveLlmSnapshot;
  busy: boolean;
  onGenerate: () => void;
  ncReady: boolean;
}) {
  return (
    <div style={llmBoxStyle}>
      <p style={{ margin: "0 0 8px", fontSize: 12, color: "#64748b" }}>
        调用后端 <code>/api/windwave/llm-report</code>（百炼 DashScope，密钥在服务端配置）。
      </p>
      {llm.error && <p style={{ color: "#b91c1c", fontSize: 12 }}>{llm.error}</p>}
      {!llm.summaryAnomaly && !llm.error && (
        <p style={{ fontSize: 13, color: "#64748b" }}>
          {ncReady ? (
            <>
              暂无 LLM 解读。
              <button type="button" style={{ marginLeft: 8 }} disabled={busy} onClick={onGenerate}>
                {busy ? "生成中…" : "生成智能解读"}
              </button>
            </>
          ) : (
            "该报告未关联 NC 路径，无法生成 LLM 解读。"
          )}
        </p>
      )}
      {llm.summaryAnomaly && (
        <div style={{ fontSize: 13, lineHeight: 1.55 }}>
          <p>
            <strong>综述</strong>
            <br />
            {llm.summaryAnomaly}
          </p>
          {llm.impact && (
            <p>
              <strong>影响</strong>
              <br />
              {llm.impact}
            </p>
          )}
          {llm.historicalAnalogy && (
            <p>
              <strong>历史类比</strong>
              <br />
              {llm.historicalAnalogy}
            </p>
          )}
          {(llm.actions?.length ?? 0) > 0 && (
            <div>
              <strong>建议</strong>
              <ol style={{ margin: "6px 0 0", paddingLeft: 20 }}>
                {llm.actions!.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function QuickLink({ to, label, desc }: { to: string; label: string; desc: string }) {
  return (
    <Link
      to={to}
      style={{
        flex: "1 1 180px",
        padding: 12,
        borderRadius: 10,
        border: "1px solid #e2e8f0",
        background: "#f8fafc",
        textDecoration: "none",
        color: "#0f172a",
      }}
    >
      <div style={{ fontWeight: 600, fontSize: 14 }}>{label}</div>
      <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>{desc}</div>
    </Link>
  );
}

const cardStyle: CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 10,
  padding: 12,
};

const preStyle: CSSProperties = {
  margin: 0,
  maxHeight: 560,
  overflow: "auto",
  fontSize: 12,
  lineHeight: 1.5,
  background: "#f8fafc",
  padding: 12,
  borderRadius: 8,
  whiteSpace: "pre-wrap",
};

const llmBoxStyle: CSSProperties = {
  maxHeight: 560,
  overflow: "auto",
  background: "#f8fafc",
  padding: 12,
  borderRadius: 8,
  border: "1px solid #e2e8f0",
};

const tabBtn = (active: boolean): CSSProperties => ({
  padding: "4px 10px",
  borderRadius: 8,
  border: "1px solid #cbd5e1",
  background: active ? "#0369a1" : "#f8fafc",
  color: active ? "#fff" : "#0f172a",
  fontSize: 12,
});

const listBtn = (active: boolean): CSSProperties => ({
  display: "block",
  width: "100%",
  textAlign: "left",
  padding: "10px 8px",
  marginBottom: 4,
  border: active ? "1px solid #0369a1" : "1px solid #e2e8f0",
  borderRadius: 8,
  background: active ? "#f0f9ff" : "#fff",
  cursor: "pointer",
});
