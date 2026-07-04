import { useCallback, useEffect, useState, type CSSProperties } from "react";
import { fetchNcMeta, postNcSubset, type NcMetaResponse } from "../adapters/preprocessAdapter";
import { uploadNcFiles } from "../adapters/ncOfflineAdapter";
import type { OceanMode } from "./offlineSession";
import { useOceanSession } from "./offlineSession";

export type TaskType = "eddy" | "windwave";

type Props = {
  mode: OceanMode;
  /** 确认裁剪时的变量校验：涡旋页 eddy、风浪页 windwave、总览不传 */
  subsetTask?: TaskType | null;
};

/** 论文 §4.5 左侧：上传 + 可选时空裁剪 */
export function TaskConfigPanel({ mode, subsetTask = null }: Props) {
  const session = useOceanSession(mode);
  const [enableSubset, setEnableSubset] = useState(false);
  const [meta, setMeta] = useState<NcMetaResponse | null>(null);
  const [timeStart, setTimeStart] = useState("");
  const [timeStop, setTimeStop] = useState("");
  const [lonMin, setLonMin] = useState("");
  const [lonMax, setLonMax] = useState("");
  const [latMin, setLatMin] = useState("");
  const [latMax, setLatMax] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [lastSubset, setLastSubset] = useState<string | null>(null);

  const loadMeta = useCallback(async (path: string) => {
    setErr(null);
    try {
      const m = await fetchNcMeta(path);
      setMeta(m);
      if (m.time_len != null && m.time_len > 0) {
        setTimeStart("0");
        setTimeStop(String(m.time_len - 1));
      }
      if (m.lon_min != null) setLonMin(String(m.lon_min));
      if (m.lon_max != null) setLonMax(String(m.lon_max));
      if (m.lat_min != null) setLatMin(String(m.lat_min));
      if (m.lat_max != null) setLatMax(String(m.lat_max));
    } catch (e) {
      setMeta(null);
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    if (mode !== "offline" || !session.ncPath) {
      if (mode === "offline") setMeta(null);
      return;
    }
    void loadMeta(session.ncPath);
  }, [mode, session.ncPath, loadMeta]);

  const onUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!/\.(nc|nc4|cdf)$/i.test(file.name)) {
      setErr("请上传 NetCDF：.nc / .nc4 / .cdf");
      e.target.value = "";
      return;
    }
    setBusy(true);
    setErr(null);
    setLastSubset(null);
    try {
      const paths = await uploadNcFiles([file]);
      const p = paths[0];
      session.setNcPath(p);
      if (enableSubset) {
        session.setPipelineArmed(false);
      } else {
        session.setPipelineArmed(true);
      }
      await loadMeta(p);
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
      e.target.value = "";
    }
  };

  const onToggleSubset = (checked: boolean) => {
    setEnableSubset(checked);
    setErr(null);
    if (!checked && session.ncPath) {
      session.setPipelineArmed(true);
    } else if (checked && session.ncPath) {
      session.setPipelineArmed(false);
    }
  };

  const parseIdx = (s: string): number | undefined => {
    const t = s.trim();
    if (!t) return undefined;
    const n = Number(t);
    return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : undefined;
  };

  const parseFloatOrUndef = (s: string): number | undefined => {
    const t = s.trim();
    if (!t) return undefined;
    const n = Number(t);
    return Number.isFinite(n) ? n : undefined;
  };

  const onConfirmSubset = async () => {
    if (!session.ncPath) {
      setErr("请先上传 NC");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const out = await postNcSubset({
        nc_path: session.ncPath,
        time_start: parseIdx(timeStart),
        time_stop: parseIdx(timeStop),
        lon_min: parseFloatOrUndef(lonMin),
        lon_max: parseFloatOrUndef(lonMax),
        lat_min: parseFloatOrUndef(latMin),
        lat_max: parseFloatOrUndef(latMax),
        task: subsetTask ?? undefined,
      });
      session.setNcPath(out.nc_path);
      session.setPipelineArmed(true);
      setLastSubset(out.nc_path);
      await loadMeta(out.nc_path);
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  };

  if (mode === "realtime") {
    return (
      <div className="task-config-panel" style={panelStyle}>
        <strong style={{ fontSize: 14 }}>任务配置</strong>
        <p style={{ margin: "8px 0 0", fontSize: 12, color: "#64748b" }}>
          准实时模式请在下方点击「拉取 latest」；裁剪 API 对接 Phase 2。
        </p>
      </div>
    );
  }

  const vm = meta?.variable_map;

  return (
    <div className="task-config-panel" style={panelStyle}>
      <strong style={{ fontSize: 14 }}>任务配置</strong>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 10, alignItems: "center" }}>
        <label style={{ fontSize: 13 }}>
          上传 NC
          <input type="file" accept=".nc,.nc4,.cdf" onChange={onUpload} disabled={busy} style={{ marginLeft: 8 }} />
        </label>
        <label style={{ fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}>
          <input
            type="checkbox"
            checked={enableSubset}
            onChange={(e) => onToggleSubset(e.target.checked)}
            disabled={busy}
          />
          应用时空裁剪
        </label>
      </div>

      {!enableSubset && (
        <p style={{ margin: "8px 0 0", fontSize: 12, color: "#64748b" }}>
          未勾选时：上传后立即作为分析输入（监测总览自动跑涡旋与风浪）。
        </p>
      )}
      {enableSubset && (
        <p style={{ margin: "8px 0 0", fontSize: 12, color: "#64748b" }}>
          已勾选：上传后仅登记文件，请设置范围并点击「确认裁剪」后再运行各模块。
        </p>
      )}

      {session.ncPath && (
        <p style={{ fontSize: 12, color: "#475569", margin: "8px 0 0" }}>
          当前 NC：<code style={{ fontSize: 11 }}>{session.ncPath}</code>
          {lastSubset ? "（已裁剪）" : ""}
          {!session.pipelineArmed && enableSubset ? " · 待确认裁剪" : ""}
        </p>
      )}

      {enableSubset && meta && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
            gap: 8,
            marginTop: 10,
          }}
        >
          <label style={labelStyle}>
            时间起索引
            <input value={timeStart} onChange={(e) => setTimeStart(e.target.value)} style={inputStyle} />
          </label>
          <label style={labelStyle}>
            时间止索引
            <input value={timeStop} onChange={(e) => setTimeStop(e.target.value)} style={inputStyle} />
          </label>
          <label style={labelStyle}>
            经度 min
            <input value={lonMin} onChange={(e) => setLonMin(e.target.value)} style={inputStyle} />
          </label>
          <label style={labelStyle}>
            经度 max
            <input value={lonMax} onChange={(e) => setLonMax(e.target.value)} style={inputStyle} />
          </label>
          <label style={labelStyle}>
            纬度 min
            <input value={latMin} onChange={(e) => setLatMin(e.target.value)} style={inputStyle} />
          </label>
          <label style={labelStyle}>
            纬度 max
            <input value={latMax} onChange={(e) => setLatMax(e.target.value)} style={inputStyle} />
          </label>
        </div>
      )}

      {meta?.time_start_label && (
        <p style={{ fontSize: 11, color: "#64748b", margin: "6px 0 0" }}>
          时间范围：{meta.time_start_label} → {meta.time_end_label}（共 {meta.time_len} 步）
        </p>
      )}

      {vm && (
        <p style={{ fontSize: 11, color: "#64748b", margin: "6px 0 0" }}>
          变量映射：涡旋 {vm.eddy_ready ? "✓" : "—"} · 风浪 {vm.windwave_ready ? "✓" : "—"}
          {vm.found?.adt ? ` · adt→${vm.found.adt}` : vm.found?.sla ? ` · sla→${vm.found.sla}` : ""}
        </p>
      )}

      {enableSubset && (
        <div style={{ marginTop: 10 }}>
          <button type="button" onClick={() => void onConfirmSubset()} disabled={busy || !session.ncPath}>
            {busy ? "处理中…" : "确认裁剪"}
          </button>
        </div>
      )}

      {err && <p style={{ color: "#b91c1c", fontSize: 12, marginTop: 8 }}>{err}</p>}
    </div>
  );
}

const panelStyle: CSSProperties = {
  padding: "12px 14px",
  marginBottom: 12,
  background: "#f8fafc",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
};

const labelStyle: CSSProperties = { fontSize: 12, display: "flex", flexDirection: "column", gap: 4 };

const inputStyle: CSSProperties = {
  padding: "4px 6px",
  fontSize: 12,
  border: "1px solid #cbd5e1",
  borderRadius: 4,
};
