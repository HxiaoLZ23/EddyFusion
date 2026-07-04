import { useMemo, useState, type CSSProperties } from "react";
import { useSearchParams } from "react-router-dom";
import {
  postEddyPreviewFrame,
  type EddyPreviewFrameResponse,
  type EddyPreviewStatRow,
} from "../adapters/eddyPreviewAdapter";
import { TaskConfigPanel } from "../dashboard/TaskConfigPanel";
import { type OceanMode, useOceanSession } from "../dashboard/offlineSession";

function contourPoints(row: EddyPreviewStatRow): string {
  const pts = row.contour_xy;
  if (pts && pts.length >= 3) {
    return pts.map(([x, y]) => `${x},${y}`).join(" ");
  }
  const bb = row.bbox_xywh;
  if (bb) {
    const [x, y, w, h] = bb;
    return `${x},${y} ${x + w},${y} ${x + w},${y + h} ${x},${y + h}`;
  }
  return "";
}

function EddyFrameOverlay({
  frame,
  selectedRow,
}: {
  frame: EddyPreviewFrameResponse;
  selectedRow: EddyPreviewStatRow | null;
}) {
  const [h, w] = frame.shape_hw;
  const pts = selectedRow ? contourPoints(selectedRow) : "";
  return (
    <div style={{ position: "relative", width: "100%" }}>
      <img
        src={frame.image_data_url}
        alt="eddy preview frame"
        style={{ width: "100%", display: "block", borderRadius: 8, border: "1px solid #cbd5e1" }}
      />
      {selectedRow && pts && (
        <svg
          aria-hidden
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
          viewBox={`0 0 ${w} ${h}`}
          preserveAspectRatio="xMidYMid meet"
        >
          <polygon points={pts} fill="rgba(251, 191, 36, 0.28)" stroke="#f59e0b" strokeWidth={Math.max(2, w / 320)} />
          <circle
            cx={selectedRow.centroid_xy[0]}
            cy={selectedRow.centroid_xy[1]}
            r={Math.max(3, w / 160)}
            fill="#dc2626"
            stroke="#fff"
            strokeWidth={1}
          />
        </svg>
      )}
    </div>
  );
}

/** Phase 2：涡旋分析页左-中-右（preview-frame + 实例属性查询）。 */
export function EddyAnalysisPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const mode: OceanMode = searchParams.get("source") === "realtime" ? "realtime" : "offline";
  const session = useOceanSession(mode);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [timeIndex, setTimeIndex] = useState(0);
  const [frame, setFrame] = useState<EddyPreviewFrameResponse | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const setSource = (m: OceanMode) => setSearchParams(m === "realtime" ? { source: "realtime" } : {});

  const canRun = !!session.ncPath && session.pipelineArmed;
  const selectedRow = useMemo(
    () => frame?.stats_rows?.find((r) => r.id === selectedId) ?? null,
    [frame, selectedId],
  );

  const summaryText = useMemo(() => {
    if (!frame?.summary) return "未生成";
    const n = frame.summary.candidate_count ?? 0;
    const conf = frame.summary.peak_conf;
    const src = frame.source === "yolo" ? "YOLO" : "ADT降级";
    return `候选 ${n}${typeof conf === "number" ? ` · peak_conf=${conf.toFixed(3)}` : ""} · ${src}`;
  }, [frame]);

  const onPreview = async () => {
    if (!session.ncPath) {
      setErr("请先在左侧上传并选择 NC");
      return;
    }
    setBusy(true);
    setErr(null);
    setSelectedId(null);
    try {
      const out = await postEddyPreviewFrame({ nc_path: session.ncPath, time_index: Math.max(0, timeIndex) });
      setFrame(out);
      const rows = out.stats_rows || [];
      const peakConf = out.summary?.peak_conf ?? 0;
      session.setEddyDetectionFrames(
        rows.length
          ? [
              {
                time: out.time_label || `step ${out.time_index}`,
                peak_score: peakConf,
                max_conf: peakConf,
                mean_conf: peakConf,
                status: rows.length > 0 ? "hit" : "miss",
                count: rows.length,
              },
            ]
          : [],
      );
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
        <h2 style={{ margin: 0 }}>涡旋识别分析</h2>
        <button type="button" onClick={() => setSource("offline")} style={tabStyle(mode === "offline")}>
          离线
        </button>
        <button type="button" onClick={() => setSource("realtime")} style={tabStyle(mode === "realtime")}>
          准实时
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr 360px", gap: 12, alignItems: "start" }}>
        <div>
          <TaskConfigPanel mode={mode} subsetTask="eddy" />
          <div style={cardStyle}>
            <strong style={{ fontSize: 14 }}>时间轴（索引）</strong>
            <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="number"
                min={0}
                value={timeIndex}
                onChange={(e) => setTimeIndex(Math.max(0, Number(e.target.value || 0)))}
                style={{ width: 90 }}
              />
              <button type="button" disabled={!canRun || busy} onClick={() => void onPreview()}>
                {busy ? "加载中…" : "加载预览帧"}
              </button>
            </div>
            <p style={{ margin: "8px 0 0", fontSize: 12, color: "#64748b" }}>
              单帧 YOLO 推理；若权重未就绪自动降级为 ADT 阈值可视化。点击右侧表格行可高亮对应实例。
              {!session.pipelineArmed && session.ncPath ? "（已勾选裁剪时请先确认裁剪）" : ""}
            </p>
          </div>
        </div>

        <div style={cardStyle}>
          <strong style={{ fontSize: 14 }}>
            中央场图（{frame?.source === "yolo" ? "YOLO mask 叠加" : frame?.source === "adt_fallback" ? "ADT 阈值降级" : "涡旋识别结果"}）
          </strong>
          <p style={{ margin: "6px 0 10px", fontSize: 12, color: "#64748b" }}>
            {frame
              ? `time_index=${frame.time_index}${frame.time_label ? ` · ${frame.time_label}` : ""}${selectedRow ? ` · 已选 #${selectedRow.id}` : ""}`
              : "未加载 · 点击【加载预览帧】"}
          </p>
          {frame?.image_data_url ? (
            <EddyFrameOverlay frame={frame} selectedRow={selectedRow} />
          ) : (
            <div style={emptyStyle}>加载后在此显示帧图与候选框叠加</div>
          )}
          {selectedRow && (
            <div style={{ marginTop: 10, fontSize: 12, color: "#334155", lineHeight: 1.6 }}>
              <strong>实例 #{selectedRow.id}</strong>
              {" · "}
              类型 {selectedRow.eddy_type ?? "—"}
              {" · "}
              面积 {selectedRow.area_px} px
              {" · "}
              周长 {typeof selectedRow.perimeter_px === "number" ? selectedRow.perimeter_px.toFixed(1) : "—"} px
              {" · "}
              质心 ({selectedRow.centroid_xy[0].toFixed(1)}, {selectedRow.centroid_xy[1].toFixed(1)})
              {" · "}
              置信度 {typeof selectedRow.confidence === "number" ? selectedRow.confidence.toFixed(3) : "—"}
            </div>
          )}
          {err && <p style={{ color: "#b91c1c", marginTop: 8, fontSize: 12 }}>{err}</p>}
        </div>

        <div style={cardStyle}>
          <strong style={{ fontSize: 14 }}>检测统计</strong>
          <p style={{ margin: "6px 0 10px", fontSize: 12, color: "#64748b" }}>{summaryText}</p>
          <div style={{ maxHeight: 520, overflow: "auto", border: "1px solid #e2e8f0", borderRadius: 8 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ background: "#f8fafc" }}>
                  <th style={thStyle}>ID</th>
                  <th style={thStyle}>类型</th>
                  <th style={thStyle}>面积(px)</th>
                  <th style={thStyle}>周长(px)</th>
                  <th style={thStyle}>中心(x,y)</th>
                  <th style={thStyle}>置信度</th>
                </tr>
              </thead>
              <tbody>
                {(frame?.stats_rows || []).map((r) => (
                  <tr
                    key={r.id}
                    onClick={() => setSelectedId((prev) => (prev === r.id ? null : r.id))}
                    style={{
                      cursor: "pointer",
                      background: selectedId === r.id ? "#fffbeb" : undefined,
                    }}
                  >
                    <td style={tdStyle}>{r.id}</td>
                    <td style={tdStyle}>{r.eddy_type ?? "—"}</td>
                    <td style={tdStyle}>{r.area_px}</td>
                    <td style={tdStyle}>{typeof r.perimeter_px === "number" ? r.perimeter_px.toFixed(1) : "—"}</td>
                    <td style={tdStyle}>
                      {r.centroid_xy[0].toFixed(1)}, {r.centroid_xy[1].toFixed(1)}
                    </td>
                    <td style={tdStyle}>{typeof r.confidence === "number" ? r.confidence.toFixed(3) : "—"}</td>
                  </tr>
                ))}
                {!frame?.stats_rows?.length && (
                  <tr>
                    <td style={tdStyle} colSpan={6}>
                      暂无候选
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
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

const emptyStyle: CSSProperties = {
  minHeight: 360,
  border: "1px dashed #cbd5e1",
  borderRadius: 8,
  display: "grid",
  placeItems: "center",
  color: "#64748b",
  fontSize: 13,
};

const thStyle: CSSProperties = { textAlign: "left", padding: "6px 8px", borderBottom: "1px solid #e2e8f0" };
const tdStyle: CSSProperties = { padding: "6px 8px", borderBottom: "1px solid #f1f5f9" };
const tabStyle = (active: boolean): CSSProperties => ({
  padding: "4px 10px",
  borderRadius: 8,
  border: "1px solid #cbd5e1",
  background: active ? "#0369a1" : "#f8fafc",
  color: active ? "#fff" : "#0f172a",
  fontSize: 12,
});
