import type { WindwaveSeriesPoint } from "../adapters/windwaveForecastAdapter";
import { levelColor } from "../adapters/windwaveForecastAdapter";

type Props = {
  series: WindwaveSeriesPoint[];
};

function scalePoints(values: number[], width: number, height: number, pad = 14): string {
  if (values.length < 1) return "";
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length < 1) return "";
  const minV = Math.min(...finite);
  const maxV = Math.max(...finite);
  const span = Math.max(maxV - minV, 1e-6);
  return values
    .map((v, i) => {
      const x = pad + (i / Math.max(values.length - 1, 1)) * (width - pad * 2);
      const y = height - pad - ((v - minV) / span) * (height - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function shadeRects(
  series: WindwaveSeriesPoint[],
  width: number,
  height: number,
  pad = 14,
): { x: number; w: number; fill: string }[] {
  if (series.length < 2) return [];
  const stepW = (width - pad * 2) / Math.max(series.length - 1, 1);
  return series.map((s, i) => ({
    x: pad + i * stepW - stepW / 2,
    w: stepW,
    fill: levelColor(s.level),
  }));
}

/** 双头 LSTM 风格曲线 + 异常时段色带（黄/橙/红） */
export function WindWaveForecastChart({ series }: Props) {
  const data = series.slice(0, 120);
  if (data.length < 2) {
    return (
      <p style={{ color: "#64748b", fontSize: 13, margin: 0 }}>
        暂无风浪时序，请上传含 u10/v10 或有效波高的 NetCDF 后运行预测。
      </p>
    );
  }
  const width = 640;
  const height = 220;
  const windObs = data.map((d) => Number(d.wind_observed));
  const windPred = data.map((d) => Number(d.wind_predicted));
  const waveObs = data.map((d) => Number(d.wave_observed));
  const wavePred = data.map((d) => Number(d.wave_predicted));
  const shades = shadeRects(data, width, height);

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="风浪预测与异常高亮" style={{ width: "100%" }}>
        <rect x="0" y="0" width={width} height={height} rx="10" fill="#f8fafc" />
        {shades.map((s, i) =>
          s.fill !== "transparent" ? (
            <rect key={i} x={Math.max(0, s.x)} y={12} width={s.w} height={height - 24} fill={s.fill} />
          ) : null,
        )}
        <polyline points={scalePoints(windObs, width, height)} fill="none" stroke="#0369a1" strokeWidth="2" />
        <polyline points={scalePoints(windPred, width, height)} fill="none" stroke="#7dd3fc" strokeWidth="1.5" strokeDasharray="4 3" />
        <polyline points={scalePoints(waveObs, width, height)} fill="none" stroke="#b45309" strokeWidth="2" />
        <polyline points={scalePoints(wavePred, width, height)} fill="none" stroke="#fdba74" strokeWidth="1.5" strokeDasharray="4 3" />
      </svg>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 8, fontSize: 12, color: "#475569" }}>
        <span><span style={{ color: "#0369a1" }}>■</span> 风速观测</span>
        <span><span style={{ color: "#7dd3fc" }}>■</span> 风速预测</span>
        <span><span style={{ color: "#b45309" }}>■</span> 浪高观测</span>
        <span><span style={{ color: "#fdba74" }}>■</span> 浪高预测</span>
        <span style={{ marginLeft: 8 }}>背景色：黄=低 · 橙=中 · 红=高</span>
      </div>
    </div>
  );
}
