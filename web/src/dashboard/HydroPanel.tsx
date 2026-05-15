import { useCallback, useEffect, useRef, useState } from "react";
import { postHydroHeatmap, type HeatmapResponse } from "../adapters/hydroHeatmapAdapter";
import { HydroHeatmapMap, type GridData } from "../map/HydroHeatmapMap";
import { HydroRotatingCurves, type CurveDataMap } from "./HydroRotatingCurves";

const AUTO_THROTTLE_MS = 5000;

type Props = {
  ncPaths: string[];
  /** 实时：路径或指纹变化时自动拉热力图（节流） */
  autoLoadOnPathChange?: boolean;
  defaultLead?: number;
};

export function HydroPanel({ ncPaths, autoLoadOnPathChange, defaultLead = 71 }: Props) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [grid, setGrid] = useState<GridData | null>(null);
  const [meta, setMeta] = useState<HeatmapResponse["meta"] | null>(null);
  const [curveData, setCurveData] = useState<CurveDataMap | null>(null);
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [lead, setLead] = useState(defaultLead);
  const [feature, setFeature] = useState("temp");
  const [kind, setKind] = useState("pred");
  const lastAutoAt = useRef(0);

  const loadHeatmap = useCallback(async () => {
    if (!ncPaths.length) {
      setErr("请先提供至少一个 NetCDF 路径");
      return;
    }
    setErr(null);
    setBusy(true);
    try {
      const res = await postHydroHeatmap({
        nc_paths: ncPaths,
        lead_time_index: lead,
        feature,
        kind,
      });
      setGrid({ lons: res.lons, lats: res.lats, values: res.values });
      setMeta(res.meta);
      setCurveData((res.curve_data as CurveDataMap) ?? null);
      setFeatureNames(Array.isArray(res.feature_names) ? res.feature_names : []);
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
      setGrid(null);
      setMeta(null);
      setCurveData(null);
      setFeatureNames([]);
    } finally {
      setBusy(false);
    }
  }, [ncPaths, lead, feature, kind]);

  const pathsKey = ncPaths.join("|");
  const loadRef = useRef(loadHeatmap);
  loadRef.current = loadHeatmap;
  useEffect(() => {
    if (!autoLoadOnPathChange || !pathsKey) return;
    const now = Date.now();
    if (now - lastAutoAt.current < AUTO_THROTTLE_MS) return;
    lastAutoAt.current = now;
    void loadRef.current();
  }, [pathsKey, autoLoadOnPathChange]);

  const insufficient = !!meta && !meta.buffer_sufficient;

  return (
    <div className="ocean-dashboard__panel ocean-dashboard__panel--hydro">
      <h3 className="ocean-dashboard__panel-head">水文区块 · 曲线（规划 §4）+ 热力图（策略 · MapLibre）</h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "flex-end", marginBottom: 8, flexShrink: 0 }}>
        <label style={{ fontSize: 12 }}>
          lead_time_index
          <input
            type="number"
            min={0}
            value={lead}
            onChange={(e) => setLead(Number(e.target.value))}
            style={{ display: "block", width: 88, marginTop: 2 }}
          />
        </label>
        <label style={{ fontSize: 12 }}>
          热力图要素
          <select value={feature} onChange={(e) => setFeature(e.target.value)} style={{ display: "block", marginTop: 2 }}>
            <option value="temp">temp</option>
            <option value="sal">sal</option>
            <option value="u">u</option>
            <option value="v">v</option>
          </select>
        </label>
        <label style={{ fontSize: 12 }}>
          热力图层 kind
          <select value={kind} onChange={(e) => setKind(e.target.value)} style={{ display: "block", marginTop: 2 }}>
            <option value="pred">pred</option>
            <option value="gt">gt</option>
            <option value="abs_err">abs_err</option>
          </select>
        </label>
        <button type="button" onClick={() => void loadHeatmap()} disabled={busy}>
          {busy ? "请求中…" : "加载水文数据"}
        </button>
      </div>
      {meta && (
        <p style={{ fontSize: 11, color: "#64748b", margin: "0 0 6px" }}>
          T_hat={meta.T_hat} / T_need={meta.T_need} {meta.buffer_sufficient ? "· 缓冲充足" : "· 缓冲不足"}
        </p>
      )}
      {err && (
        <p style={{ color: "#b91c1c", fontSize: 12, margin: "0 0 6px" }}>
          {err}
        </p>
      )}
      <div className="ocean-dashboard__hydro-split">
        <div className="ocean-dashboard__hydro-col ocean-dashboard__hydro-col--curve">
          <HydroRotatingCurves curveData={curveData} featureNames={featureNames} insufficient={insufficient} />
        </div>
        <div className="ocean-dashboard__hydro-col ocean-dashboard__hydro-col--map">
          <HydroHeatmapMap data={grid} insufficient={!!grid && insufficient} mapHeight={260} />
        </div>
      </div>
    </div>
  );
}
