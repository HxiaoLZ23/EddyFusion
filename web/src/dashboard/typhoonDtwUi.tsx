import type { CSSProperties } from "react";

export type TyphoonDtwMeta = {
  enabled?: boolean;
  match_mode?: string;
  query_curve?: string;
  history_curve?: string;
  normalized?: boolean;
  n_candidates_with_track?: number;
  n_candidates_peak_fallback?: number;
  reason?: string;
  fallback_reason?: string | null;
  window?: {
    t_start?: number;
    t_end?: number;
    t_start_padded?: number;
    t_end_padded?: number;
    fallback_reason?: string | null;
    tau?: number;
  };
};

export type WindTrackCarrier = {
  wind_track_mps?: number[];
  wind_track_kt?: number[];
  series_source?: string;
};

export function parseTyphoonDtwMeta(retrieval: unknown): TyphoonDtwMeta | null {
  if (!retrieval || typeof retrieval !== "object") return null;
  const dtw = (retrieval as { dtw?: unknown }).dtw;
  if (!dtw || typeof dtw !== "object") return null;
  return dtw as TyphoonDtwMeta;
}

export function windTrackValues(c: WindTrackCarrier): number[] {
  const mps = c.wind_track_mps;
  if (Array.isArray(mps) && mps.length > 0) {
    return mps.map((v) => Number(v)).filter((v) => Number.isFinite(v));
  }
  const kt = c.wind_track_kt;
  if (Array.isArray(kt) && kt.length > 0) {
    return kt.map((v) => Number(v) * 0.514444).filter((v) => Number.isFinite(v));
  }
  return [];
}

function sparklinePoints(values: number[], width: number, height: number, pad: number): string {
  if (values.length < 1) return "";
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const span = Math.max(maxV - minV, 1e-6);
  return values
    .map((v, i) => {
      const x = pad + (i / Math.max(values.length - 1, 1)) * (width - pad * 2);
      const y = height - pad - ((v - minV) / span) * (height - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function WindTrackSparkline({
  values,
  width = 88,
  height = 22,
  title,
}: {
  values: number[];
  width?: number;
  height?: number;
  title?: string;
}) {
  if (values.length < 2) {
    return (
      <span style={{ fontSize: 11, color: "#94a3b8" }} title={title}>
        无轨迹
      </span>
    );
  }
  const pts = sparklinePoints(values, width, height, 2);
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={title ?? "IBTrACS 中心风速轨迹"}
      style={{ display: "block", verticalAlign: "middle" }}
    >
      <rect x={0} y={0} width={width} height={height} rx={3} fill="#f8fafc" stroke="#e2e8f0" />
      <polyline points={pts} fill="none" stroke="#2563eb" strokeWidth={1.4} strokeLinejoin="round" />
    </svg>
  );
}

const metaBarStyle: CSSProperties = {
  margin: "0 0 10px",
  padding: "8px 10px",
  borderRadius: 8,
  background: "#f8fafc",
  border: "1px solid #e2e8f0",
  fontSize: 11,
  color: "#475569",
  lineHeight: 1.5,
};

export type TyphoonQueryMeta = {
  start_time?: string;
  end_time?: string;
  anomaly_start_time?: string;
  anomaly_end_time?: string;
  history_search_mode?: string;
  history_lookback_years?: number;
  lon_min?: number;
  lon_max?: number;
  lat_min?: number;
  lat_max?: number;
};

export function formatHistorySearchLabel(query: TyphoonQueryMeta | null | undefined): string | null {
  if (!query) return null;
  const mode = query.history_search_mode ?? "full";
  if (mode === "lookback") {
    const years = query.history_lookback_years ?? 10;
    return `历史检索：向前 ${years} 年（${query.start_time ?? "—"} ~ ${query.end_time ?? "—"}）`;
  }
  return `历史检索：全库（${query.start_time ?? "—"} ~ ${query.end_time ?? "—"}）`;
}

export function TyphoonDtwMetaBar({
  meta,
  query,
}: {
  meta: TyphoonDtwMeta | null;
  query?: TyphoonQueryMeta | null;
}) {
  if (!meta) return null;
  if (meta.enabled === false) {
    return (
      <div style={metaBarStyle}>
        <strong>DTW</strong>：未启用
        {meta.reason ? `（${meta.reason}）` : ""}
      </div>
    );
  }
  const mode = meta.match_mode ?? "—";
  const withTrack = meta.n_candidates_with_track ?? "—";
  const fallback = meta.n_candidates_peak_fallback ?? 0;
  const queryLabel =
    meta.query_curve === "wind_obs_regional_mean_window"
      ? "异常窗内区域平均风速观测"
      : meta.query_curve ?? "wind_dtw_curve";
  return (
    <div style={metaBarStyle}>
      <div>
        <strong>DTW 口径</strong>：<code style={{ fontSize: 10 }}>{mode}</code>
      </div>
      <div>
        风轨迹候选 <strong>{withTrack}</strong> 条
        {typeof fallback === "number" && fallback > 0 ? ` · 峰值常数降级 ${fallback} 条` : ""}
        {meta.normalized ? " · z-score 形态匹配" : ""}
      </div>
      <div style={{ color: "#64748b" }}>
        查询：{queryLabel} → 历史：{meta.history_curve ?? "IBTrACS 中心风速 wind_track_mps"}
      </div>
      {meta.fallback_reason && (
        <div style={{ color: "#64748b" }}>异常窗 fallback：{meta.fallback_reason}</div>
      )}
      {meta.window &&
        typeof meta.window.t_start_padded === "number" &&
        typeof meta.window.t_end_padded === "number" && (
          <div style={{ color: "#64748b" }}>
            事件窗步索引（含 padding）：{meta.window.t_start_padded} ~ {meta.window.t_end_padded}
          </div>
        )}
      {query?.anomaly_start_time && query?.anomaly_end_time && (
        <div style={{ color: "#64748b" }}>
          当前异常时段：{query.anomaly_start_time} ~ {query.anomaly_end_time}
        </div>
      )}
      {formatHistorySearchLabel(query) && (
        <div style={{ color: "#64748b" }}>{formatHistorySearchLabel(query)}</div>
      )}
    </div>
  );
}

export function formatMatchModeShort(mode: string | undefined): string {
  if (mode === "wind_residual_vs_ibtracs_track") return "风残差 ↔ IBTrACS 中心风（legacy）";
  if (mode === "regional_mean_obs_vs_ibtracs_center") return "区域平均观测风 ↔ IBTrACS 中心风";
  return mode ?? "—";
}
