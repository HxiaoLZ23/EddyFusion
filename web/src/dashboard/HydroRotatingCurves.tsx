import { useEffect, useMemo, useState } from "react";

export type CurvePoint = { horizon: number; gt: number; pred: number };
export type CurveDataMap = Record<string, CurvePoint[]>;

const ROTATE_MS = 2000;

type Props = {
  curveData: CurveDataMap | null;
  featureNames: string[];
  featureUnits?: Record<string, string>;
  /** 缓冲不足：曲线区虚化 + 蒙层（与规划 §4 一致） */
  insufficient: boolean;
};

function axisTicks(min: number, max: number, n: number): number[] {
  if (!(Number.isFinite(min) && Number.isFinite(max))) return [0];
  if (max < min) {
    const t = min;
    min = max;
    max = t;
  }
  if (Math.abs(max - min) < 1e-15) return [min];
  const out: number[] = [];
  const steps = Math.max(2, n);
  for (let i = 0; i < steps; i++) {
    out.push(min + ((max - min) * i) / (steps - 1));
  }
  return out;
}

type ChartGeom = {
  w: number;
  h: number;
  padL: number;
  padR: number;
  padT: number;
  padB: number;
  lineGt: string;
  linePd: string;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  xTicks: number[];
  yTicks: number[];
  sx: (x: number) => number;
  sy: (y: number) => number;
};

export function HydroRotatingCurves({ curveData, featureNames, featureUnits, insufficient }: Props) {
  const names = useMemo(() => featureNames.filter((n) => curveData?.[n]?.length), [featureNames, curveData]);
  const [idx, setIdx] = useState(0);
  const [hoverPause, setHoverPause] = useState(false);

  useEffect(() => {
    setIdx(0);
  }, [names.join("|")]);

  useEffect(() => {
    if (!names.length || hoverPause || insufficient) return;
    const id = window.setInterval(() => {
      setIdx((i) => (i + 1) % names.length);
    }, ROTATE_MS);
    return () => window.clearInterval(id);
  }, [names.length, names.join("|"), hoverPause, insufficient]);

  const current = names.length ? names[idx % names.length]! : "";
  const series = current && curveData ? curveData[current] ?? [] : [];

  const chart = useMemo((): ChartGeom | null => {
    if (!series.length) return null;
    const w = 440;
    const h = 200;
    const padL = 52;
    const padR = 18;
    const padT = 36;
    const padB = 44;
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
    const dy = yMax - yMin || 1;
    const iw = w - padL - padR;
    const ih = h - padT - padB;
    const sx = (x: number) => padL + ((x - xMin) / dx) * iw;
    const sy = (y: number) => padT + (1 - (y - yMin) / dy) * ih;
    const lineGt = series.map((p) => `${sx(p.horizon)},${sy(p.gt)}`).join(" ");
    const linePd = series.map((p) => `${sx(p.horizon)},${sy(p.pred)}`).join(" ");
    const xTicks = axisTicks(xMin, xMax, 6);
    const yTicks = axisTicks(yMin, yMax, 5);
    return {
      w,
      h,
      padL,
      padR,
      padT,
      padB,
      lineGt,
      linePd,
      xMin,
      xMax,
      yMin,
      yMax,
      xTicks,
      yTicks,
      sx,
      sy,
    };
  }, [series]);

  const fmtY = (v: number) => {
    const a = Math.abs(v);
    if (a >= 1000 || (a > 0 && a < 1e-3)) return v.toExponential(2);
    if (a >= 1) return v.toFixed(3);
    return v.toFixed(4);
  };

  const fmtX = (v: number) => (Math.abs(v - Math.round(v)) < 1e-5 ? String(Math.round(v)) : v.toFixed(2));

  return (
    <div
      className="hydro-curves"
      style={{ position: "relative", flex: "1 1 auto", minHeight: 200, height: "100%" }}
      onMouseEnter={() => setHoverPause(true)}
      onMouseLeave={() => setHoverPause(false)}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: "#334155" }}>
          区域均值曲线（轮换 {ROTATE_MS / 1000}s{hoverPause ? " · 已悬停暂停" : ""}）
        </span>
        {current && (
          <span style={{ fontSize: 12, color: "#0369a1", fontWeight: 600 }}>{current}</span>
        )}
      </div>
      <div
        style={{
          borderRadius: 8,
          border: "1px solid #e2e8f0",
          background: "#f8fafc",
          opacity: insufficient ? 0.45 : 1,
          filter: insufficient ? "blur(1px)" : undefined,
          overflow: "hidden",
        }}
      >
        {!chart || !names.length ? (
          <div style={{ height: 200, display: "flex", alignItems: "center", justifyContent: "center", color: "#94a3b8", fontSize: 13 }}>
            加载热力图后将显示与单模块同源的 curve_data（gt / pred）
          </div>
        ) : (
          <svg width="100%" height={220} viewBox={`0 0 ${chart.w} ${chart.h}`} preserveAspectRatio="xMidYMid meet" className="hydro-curves__svg">
            <rect x={0} y={0} width={chart.w} height={chart.h} fill="#fafafa" />
            {/* 图例：置于绘图区上方，与曲线颜色一致，避免与轴区错位 */}
            <g transform={`translate(${chart.padL}, 10)`} aria-label="legend">
              <line x1={0} y1={0} x2={22} y2={0} stroke="#0369a1" strokeWidth={2.5} />
              <text x={28} y={4} fontSize={11} fill="#0f172a" fontWeight={600}>
                gt（真值）
              </text>
              <line x1={100} y1={0} x2={122} y2={0} stroke="#ea580c" strokeWidth={2.5} strokeDasharray="5 3" />
              <text x={128} y={4} fontSize={11} fill="#0f172a" fontWeight={600}>
                pred（预测）
              </text>
            </g>
            {/* 绘图区边框 */}
            <line
              x1={chart.padL}
              y1={chart.padT}
              x2={chart.padL}
              y2={chart.h - chart.padB}
              stroke="#94a3b8"
              strokeWidth={1}
            />
            <line
              x1={chart.padL}
              y1={chart.h - chart.padB}
              x2={chart.w - chart.padR}
              y2={chart.h - chart.padB}
              stroke="#94a3b8"
              strokeWidth={1}
            />
            <line
              x1={chart.w - chart.padR}
              y1={chart.padT}
              x2={chart.w - chart.padR}
              y2={chart.h - chart.padB}
              stroke="#cbd5e1"
              strokeWidth={1}
            />
            <line
              x1={chart.padL}
              y1={chart.padT}
              x2={chart.w - chart.padR}
              y2={chart.padT}
              stroke="#cbd5e1"
              strokeWidth={1}
            />
            {/* Y 刻度 */}
            {chart.yTicks.map((yv) => (
              <g key={`yt-${yv}`}>
                <line
                  x1={chart.padL - 4}
                  y1={chart.sy(yv)}
                  x2={chart.padL}
                  y2={chart.sy(yv)}
                  stroke="#64748b"
                  strokeWidth={1}
                />
                <text
                  x={chart.padL - 8}
                  y={chart.sy(yv) + 4}
                  fontSize={10}
                  fill="#475569"
                  textAnchor="end"
                >
                  {fmtY(yv)}
                </text>
              </g>
            ))}
            {/* X 刻度 */}
            {chart.xTicks.map((xv) => (
              <g key={`xt-${xv}`}>
                <line
                  x1={chart.sx(xv)}
                  y1={chart.h - chart.padB}
                  x2={chart.sx(xv)}
                  y2={chart.h - chart.padB + 5}
                  stroke="#64748b"
                  strokeWidth={1}
                />
                <text
                  x={chart.sx(xv)}
                  y={chart.h - chart.padB + 18}
                  fontSize={10}
                  fill="#475569"
                  textAnchor="middle"
                >
                  {fmtX(xv)}
                </text>
              </g>
            ))}
            {/* 轴线标题 */}
            <text
              x={chart.padL + (chart.w - chart.padL - chart.padR) / 2}
              y={chart.h - 6}
              fontSize={11}
              fill="#334155"
              fontWeight={600}
              textAnchor="middle"
            >
              预报步长（lead_time_index / 曲线横轴）
            </text>
            <text
              x={18}
              y={chart.padT + (chart.h - chart.padT - chart.padB) / 2}
              dominantBaseline="middle"
              fontSize={11}
              fill="#334155"
              fontWeight={600}
              textAnchor="middle"
              transform={`rotate(-90, 18, ${chart.padT + (chart.h - chart.padT - chart.padB) / 2})`}
            >
              区域均值（{current}
              {featureUnits?.[current] ? ` · ${featureUnits[current]}` : ""}）
            </text>
            <polyline fill="none" stroke="#0369a1" strokeWidth={2.2} points={chart.lineGt} />
            <polyline fill="none" stroke="#ea580c" strokeWidth={2.2} strokeDasharray="5 3" points={chart.linePd} />
          </svg>
        )}
      </div>
      {insufficient && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            top: 22,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(248,250,252,0.75)",
            color: "#0f172a",
            fontSize: 14,
            fontWeight: 600,
            borderRadius: 8,
            pointerEvents: "none",
          }}
        >
          数据不足，等待中
        </div>
      )}
    </div>
  );
}
