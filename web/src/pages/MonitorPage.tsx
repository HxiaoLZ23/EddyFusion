import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { fetchRealtimeStatus, type RealtimeStatus } from "../adapters/ncRealtimeAdapter";
import { OceanDashboard } from "../dashboard/OceanDashboard";
import { TaskConfigPanel } from "../dashboard/TaskConfigPanel";
import type { OceanMode } from "../dashboard/offlineSession";

function QuickEntry({ to, title, desc }: { to: string; title: string; desc: string }) {
  return (
    <Link
      to={to}
      style={{
        padding: "10px 12px",
        borderRadius: 10,
        border: "1px solid #e2e8f0",
        background: "#fff",
        textDecoration: "none",
        color: "#0f172a",
      }}
    >
      <div style={{ fontWeight: 600, fontSize: 13 }}>{title}</div>
      <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>{desc}</div>
    </Link>
  );
}

const tabStyle = (active: boolean) => ({
  padding: "0.35rem 0.75rem",
  borderRadius: 6,
  textDecoration: "none" as const,
  color: active ? "#fff" : "#0f172a",
  background: active ? "#0369a1" : "#e2e8f0",
  fontWeight: active ? 600 : 400,
  fontSize: 13,
});

type MonitorPageProps = {
  /** 由布局 keep-alive 传入：非当前路由时暂停准实时轮询 */
  active?: boolean;
};

/** 论文 §4.5 监测总览：数据源（离线/实时）在此切换，非顶栏主入口 */
export function MonitorPage({ active = true }: MonitorPageProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const source: OceanMode = searchParams.get("source") === "realtime" ? "realtime" : "offline";
  const [rtStatus, setRtStatus] = useState<RealtimeStatus | null>(null);

  const setSource = (mode: OceanMode) => {
    setSearchParams(mode === "realtime" ? { source: "realtime" } : {});
  };

  useEffect(() => {
    if (source !== "realtime") {
      setRtStatus(null);
      return;
    }
    void fetchRealtimeStatus().then(setRtStatus).catch(() => setRtStatus(null));
  }, [source]);

  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 12,
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontSize: 13, color: "#475569", fontWeight: 600 }}>数据源</span>
        <button type="button" style={tabStyle(source === "offline")} onClick={() => setSource("offline")}>
          离线 NC 上传
        </button>
        <button type="button" style={tabStyle(source === "realtime")} onClick={() => setSource("realtime")}>
          准实时 latest
        </button>
        <span style={{ fontSize: 12, color: "#64748b" }}>
          涡旋推理固定 <strong>3ch（ADT+ADT+ADT）</strong> · 默认上传即分析；勾选裁剪后需「确认裁剪」
        </span>
      </div>
      {source === "realtime" && rtStatus && (
        <div
          style={{
            marginBottom: 10,
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #e2e8f0",
            background: rtStatus.ready ? "#f0fdf4" : "#fff7ed",
            fontSize: 12,
            color: "#475569",
          }}
        >
          <strong>准实时连接器</strong> · {rtStatus.connected ? "已连接" : "未连接"} · 目录{" "}
          <code>{rtStatus.poll_dir}</code> · {rtStatus.nc_count ?? 0} 个 NC
          {rtStatus.latest?.filename ? ` · 最新 ${rtStatus.latest.filename}（${rtStatus.latest.mtime_iso}）` : ""}
          {rtStatus.latest?.stale ? " · 文件已陈旧" : ""}
        </div>
      )}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10, marginBottom: 12 }}>
        <QuickEntry to={`/eddy${source === "realtime" ? "?source=realtime" : ""}`} title="涡旋分析" desc="YOLO 单帧 · 双路 MP4" />
        <QuickEntry to={`/windwave${source === "realtime" ? "?source=realtime" : ""}`} title="风浪分析" desc="LSTM 曲线 · DTW Top-K" />
        <QuickEntry to="/reports" title="报告管理" desc="历史报告 · 再导出" />
        <QuickEntry to={`/${source === "realtime" ? "realtime" : "offline"}/typhoon-kb`} title="台风事件库" desc="DTW 深度检索" />
      </div>
      <TaskConfigPanel mode={source} />
      <OceanDashboard mode={source} hideOfflineUploadInBar={source === "offline"} active={active} />
    </div>
  );
}
