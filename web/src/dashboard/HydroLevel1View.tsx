import { useEffect, useMemo, useRef, useState } from "react";
import { postHydroHeatmap } from "../adapters/hydroHeatmapAdapter";
import { postHydroMeta } from "../adapters/hydroBufferAdapter";
import { uploadNcFiles } from "../adapters/ncOfflineAdapter";
import { HydroHeatmapMap } from "../map/HydroHeatmapMap";
import type { CurveDataMap, CurvePoint } from "./HydroRotatingCurves";
import { type OceanMode, useOceanSession } from "./offlineSession";

const CHART_ORDER = ["temp", "sal", "u", "v"];
const AUTO_THROTTLE_MS = 5000;
const DEFAULT_LEAD = 71;
const DEFAULT_FEATURE = "temp";
const DEFAULT_KIND = "pred";

function miniChart(series: CurvePoint[], w: number, h: number) {
  if (!series.length) return null;
  const padL = 28;
  const padR = 6;
  const padT = 8;
  const padB = 18;
  const xs = series.map((p) => p.horizon);
  const ys = series.flatMap((p) => [p.gt, p.pred]);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMin === yMax) {
    yMin -= 1e-6;
    yMax += 1e-6;
  }
  const dx = xMax - xMin || 1;
  const iw = w - padL - padR;
  const ih = h - padT - padB;
  const sx = (x: number) => padL + ((x - xMin) / dx) * iw;
  const sy = (y: number) => padT + (1 - (y - yMin) / (yMax - yMin || 1)) * ih;
  const lineGt = series.map((p) => `${sx(p.horizon)},${sy(p.gt)}`).join(" ");
  const linePd = series.map((p) => `${sx(p.horizon)},${sy(p.pred)}`).join(" ");
  return { w, h, padL, padT, padB, lineGt, linePd };
}

type Props = {
  mode: OceanMode;
  curveData: CurveDataMap | null;
  featureNames: string[];
};

function shortName(path: string): string {
  const p = path.replace(/\\/g, "/");
  const i = p.lastIndexOf("/");
  return i >= 0 ? p.slice(i + 1) : p;
}

/** 规划 §6.3：水文 L1 — 缓冲管理 + 2×2 曲线 + 热力图 */
export function HydroLevel1View({ mode, curveData, featureNames }: Props) {
  const {
    hydroBufferPaths,
    setHydroBufferPaths,
    removeHydroBufferAt,
    clearHydroBuffer,
    hydro,
    setHydro,
  } = useOceanSession(mode);
  const [maxBuf, setMaxBuf] = useState(64);
  const [meta, setMeta] = useState<{ T_need: number; T_hat: number; buffer_sufficient: boolean } | null>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [hmErr, setHmErr] = useState<string | null>(null);
  const lastAutoAt = useRef(0);
  const grid = hydro.heatmap;

  useEffect(() => {
    if (!hydroBufferPaths.length) {
      setMeta(null);
      setMetaErr(null);
      return;
    }
    let cancelled = false;
    void postHydroMeta(hydroBufferPaths)
      .then((m) => {
        if (!cancelled) {
          setMeta(m);
          setMetaErr(null);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setMetaErr(e instanceof Error ? e.message : String(e));
          setMeta(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [hydroBufferPaths]);

  useEffect(() => {
    const pathsKey = hydroBufferPaths.join("|");
    if (!pathsKey) return;
    const now = Date.now();
    if (now - lastAutoAt.current < AUTO_THROTTLE_MS) return;
    lastAutoAt.current = now;
    let cancelled = false;
    setHmErr(null);
    void postHydroHeatmap({
      nc_paths: hydroBufferPaths,
      lead_time_index: DEFAULT_LEAD,
      feature: DEFAULT_FEATURE,
      kind: DEFAULT_KIND,
    })
      .then((res) => {
        if (cancelled) return;
        const cd = (res.curve_data as CurveDataMap) ?? null;
        const fn = Array.isArray(res.feature_names) ? res.feature_names : [];
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
            value_unit: res.value_unit,
            vmin: res.vmin,
            vmax: res.vmax,
          },
        });
      })
      .catch((ex) => {
        if (!cancelled) setHmErr(ex instanceof Error ? ex.message : String(ex));
      });
    return () => {
      cancelled = true;
    };
  }, [hydroBufferPaths, setHydro]);

  const chartNames = useMemo(() => {
    const fromData = featureNames.filter((n) => curveData?.[n]?.length);
    const ordered = CHART_ORDER.filter((n) => fromData.includes(n));
    const rest = fromData.filter((n) => !CHART_ORDER.includes(n));
    return [...ordered, ...rest].slice(0, 4);
  }, [featureNames, curveData]);

  const onAddNc = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const list = e.target.files;
    if (!list?.length) return;
    setUploadBusy(true);
    setMetaErr(null);
    try {
      const paths = await uploadNcFiles(Array.from(list));
      let next = [...hydroBufferPaths];
      for (const p of paths) {
        if (!next.includes(p)) next.push(p);
      }
      while (next.length > maxBuf) next.shift();
      setHydroBufferPaths(next);
    } catch (ex) {
      setMetaErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setUploadBusy(false);
      e.target.value = "";
    }
  };

  return (
    <div className="ocean-dashboard__l1-hydro">
      <section className="ocean-dashboard__hydro-buffer">
        <h4 className="ocean-dashboard__l1-hydro-title">水文缓冲区（多文件拼接）</h4>
        <p className="ocean-dashboard__l1-hydro-note">
          按时间顺序追加日文件；缓冲充足后自动刷新曲线与热力图（默认可视化 temp / pred / lead=71）。
        </p>
        <div className="ocean-dashboard__hydro-buffer-controls">
          <label style={{ fontSize: 12 }}>
            最多保留文件数（FIFO）
            <input
              type="number"
              min={4}
              max={128}
              value={maxBuf}
              onChange={(ev) => setMaxBuf(Number(ev.target.value))}
              style={{ display: "block", width: 72, marginTop: 2 }}
            />
          </label>
          <label style={{ fontSize: 12 }}>
            追加 NC
            <input
              type="file"
              accept=".nc,.nc4,.cdf"
              multiple
              disabled={uploadBusy}
              onChange={onAddNc}
              style={{ display: "block", marginTop: 2, maxWidth: 220 }}
            />
          </label>
          <button type="button" className="ocean-dashboard__l1-back" onClick={clearHydroBuffer}>
            清空缓冲
          </button>
        </div>
        <div className="ocean-dashboard__hydro-buffer-metrics">
          <span>文件数: {hydroBufferPaths.length}</span>
          {meta && (
            <>
              <span>T_need: {meta.T_need}</span>
              <span>T̂: {meta.T_hat}</span>
              <span>{meta.buffer_sufficient ? "可构建滑窗" : "尚不足"}</span>
            </>
          )}
        </div>
        {metaErr && <p className="ocean-dashboard__hydro-buffer-err">{metaErr}</p>}
        {hydroBufferPaths.length > 0 && (
          <ul className="ocean-dashboard__hydro-buffer-list">
            {hydroBufferPaths.map((p, i) => (
              <li key={p}>
                <span title={p}>{i + 1}. {shortName(p)}</span>
                <button type="button" onClick={() => removeHydroBufferAt(i)}>
                  移除
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <h4 className="ocean-dashboard__l1-hydro-title">要素曲线（2×2）+ 热力图</h4>
      {!curveData || !chartNames.length ? (
        <p className="ocean-dashboard__l1-muted">同屏加载水文数据成功后，此处显示各要素 gt / pred 曲线与热力图。</p>
      ) : (
        <div className="ocean-dashboard__l1-hydro-viz">
          <div className="ocean-dashboard__l1-hydro-curves-grid">
            {chartNames.map((name) => {
              const series = curveData[name] ?? [];
              const g = miniChart(series, 320, 160);
              if (!g) return null;
              return (
                <div key={name} className="ocean-dashboard__l1-hydro-cell">
                  <div className="ocean-dashboard__l1-hydro-cell-head">
                    {name}
                    {hydro.featureUnits?.[name] ? ` (${hydro.featureUnits[name]})` : ""}
                  </div>
                  <svg width="100%" height={168} viewBox={`0 0 ${g.w} ${g.h}`} preserveAspectRatio="xMidYMid meet">
                    <rect width={g.w} height={g.h} fill="#fafafa" />
                    <line x1={g.padL} y1={g.h - g.padB} x2={g.w - 6} y2={g.h - g.padB} stroke="#94a3b8" strokeWidth={1} />
                    <line x1={g.padL} y1={g.padT} x2={g.padL} y2={g.h - g.padB} stroke="#94a3b8" strokeWidth={1} />
                    <polyline fill="none" stroke="#0369a1" strokeWidth={1.8} points={g.lineGt} />
                    <polyline fill="none" stroke="#ea580c" strokeWidth={1.8} strokeDasharray="4 2" points={g.linePd} />
                  </svg>
                </div>
              );
            })}
          </div>

          <section className="ocean-dashboard__l1-hydro-heatmap-panel">
            {hmErr && <p className="ocean-dashboard__hydro-buffer-err">{hmErr}</p>}
            <div className="ocean-dashboard__l1-hydro-heatmap-map">
              <HydroHeatmapMap
                data={grid ? { lons: grid.lons, lats: grid.lats, values: grid.values } : null}
                insufficient={false}
                mapHeight={320}
                vmin={grid?.vmin}
                vmax={grid?.vmax}
                unit={grid?.value_unit ?? ""}
                kind={grid?.kind ?? DEFAULT_KIND}
              />
            </div>
            {grid && (
              <p className="ocean-dashboard__l1-muted">
                当前：{grid.feature} / {grid.kind} · lead={grid.lead}
                {grid.value_unit ? ` · ${grid.value_unit}` : ""}
              </p>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
