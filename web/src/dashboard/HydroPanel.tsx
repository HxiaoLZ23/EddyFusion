import { useCallback, useEffect, useRef, useState } from "react";
import { postHydroHeatmap, type HeatmapResponse } from "../adapters/hydroHeatmapAdapter";
import { HydroHeatmapMap, type GridData } from "../map/HydroHeatmapMap";
import { type OceanMode, useOceanSession } from "./offlineSession";
import { HydroRotatingCurves, type CurveDataMap } from "./HydroRotatingCurves";

const AUTO_THROTTLE_MS = 5000;
const DEFAULT_LEAD = 71;
const DEFAULT_FEATURE = "temp";
const DEFAULT_KIND = "pred";

type Props = {
  mode: OceanMode;
  ncPaths: string[];
  /** 路径变化时自动拉水文（节流） */
  autoLoadOnPathChange?: boolean;
};

export function HydroPanel({ mode, ncPaths, autoLoadOnPathChange }: Props) {
  const { setHydro } = useOceanSession(mode);
  const [err, setErr] = useState<string | null>(null);
  const [grid, setGrid] = useState<GridData | null>(null);
  const [meta, setMeta] = useState<HeatmapResponse["meta"] | null>(null);
  const [curveData, setCurveData] = useState<CurveDataMap | null>(null);
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [featureUnits, setFeatureUnits] = useState<Record<string, string>>({});
  const [hmScale, setHmScale] = useState<{ vmin: number; vmax: number; unit: string } | null>(null);
  const lastAutoAt = useRef(0);

  const loadHeatmap = useCallback(async () => {
    if (!ncPaths.length) {
      setErr("请先提供至少一个 NetCDF 路径");
      return;
    }
    setErr(null);
    try {
      const res = await postHydroHeatmap({
        nc_paths: ncPaths,
        lead_time_index: DEFAULT_LEAD,
        feature: DEFAULT_FEATURE,
        kind: DEFAULT_KIND,
      });
      const cd = (res.curve_data as CurveDataMap) ?? null;
      const fn = Array.isArray(res.feature_names) ? res.feature_names : [];
      setGrid({ lons: res.lons, lats: res.lats, values: res.values });
      setMeta(res.meta);
      setCurveData(cd);
      setFeatureNames(fn);
      setFeatureUnits(res.feature_units ?? {});
      const unit = res.value_unit ?? "";
      const vmin = res.vmin;
      const vmax = res.vmax;
      setHmScale(
        vmin != null && vmax != null && Number.isFinite(vmin) && Number.isFinite(vmax)
          ? { vmin, vmax, unit }
          : null,
      );
      setHydro({
        curveData: cd,
        featureNames: fn,
        featureUnits: res.feature_units ?? {},
        meta: res.meta ?? null,
        heatmap: {
          lons: res.lons,
          lats: res.lats,
          values: res.values,
          feature: DEFAULT_FEATURE,
          kind: DEFAULT_KIND,
          lead: DEFAULT_LEAD,
          value_unit: unit,
          vmin,
          vmax,
        },
      });
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
      setGrid(null);
      setMeta(null);
      setCurveData(null);
      setFeatureNames([]);
      setFeatureUnits({});
      setHmScale(null);
      setHydro({ curveData: null, featureNames: [], meta: null, heatmap: null });
    }
  }, [ncPaths, setHydro]);

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
      <h3 className="ocean-dashboard__panel-head">水文区块 · 曲线 + 热力图</h3>
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
          <HydroRotatingCurves
            curveData={curveData}
            featureNames={featureNames}
            featureUnits={featureUnits}
            insufficient={insufficient}
          />
        </div>
        <div className="ocean-dashboard__hydro-col ocean-dashboard__hydro-col--map">
          <HydroHeatmapMap
            data={grid}
            insufficient={!!grid && insufficient}
            mapHeight={260}
            vmin={hmScale?.vmin}
            vmax={hmScale?.vmax}
            unit={hmScale?.unit ?? featureUnits[DEFAULT_FEATURE] ?? ""}
            kind={DEFAULT_KIND}
          />
        </div>
      </div>
    </div>
  );
}
