import { apiUrl, formatApiFetchError } from "./apiBase";

export type WindwaveSeriesPoint = {
  step: number;
  time: string;
  wind_observed: number;
  wind_predicted: number;
  wave_observed: number;
  wave_predicted: number;
  anomaly_index?: number | null;
  level?: string;
};

export type AnomalySegment = {
  step: number;
  anomaly_index: number;
  level: string;
  wind_residual: number;
  wave_residual: number;
};

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

export type TyphoonCandidate = {
  event_id?: string;
  id?: string;
  name?: string;
  start_time?: string;
  end_time?: string;
  score?: number;
  dtw_distance?: number;
  wind_track_mps?: number[];
  wind_track_kt?: number[];
  series_source?: string;
  peak_wind_kt?: number;
  [key: string]: unknown;
};

export type TyphoonRetrieval = {
  method?: string;
  dtw?: TyphoonDtwMeta;
};

export type WindwaveForecastResponse = {
  status: string;
  nc_path?: string;
  times: string[];
  wind_obs: number[];
  wind_pred: number[];
  swh_obs: number[];
  swh_pred: number[];
  series: WindwaveSeriesPoint[];
  anomaly_segments: AnomalySegment[];
  anomaly_level?: string;
  anomaly_index?: number;
  assessment_note?: string;
  typhoon_linked?: boolean;
  typhoon_link_note?: string;
    typhoon_query?: {
    start_time?: string;
    end_time?: string;
    anomaly_start_time?: string;
    anomaly_end_time?: string;
    nc_coverage_start_time?: string;
    nc_coverage_end_time?: string;
    history_search_mode?: string;
    history_lookback_years?: number;
    lon_min?: number;
    lon_max?: number;
    lat_min?: number;
    lat_max?: number;
  };
  typhoon_candidates?: TyphoonCandidate[];
  typhoon_retrieval?: TyphoonRetrieval;
  typhoon_kb_ready?: boolean;
};

export type StructuredReportResponse = {
  status: string;
  format: string;
  markdown: string;
  fields: Record<string, unknown>;
  download_name?: string;
};

export async function postWindwaveForecast(
  ncPath: string,
  topK = 5,
): Promise<WindwaveForecastResponse> {
  let res: Response;
  try {
    res = await fetch(apiUrl("/api/windwave/forecast"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nc_path: ncPath, top_k: topK }),
    });
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : res.statusText;
    throw new Error(msg);
  }
  return res.json() as Promise<WindwaveForecastResponse>;
}

export async function postStructuredReport(
  ncPath: string,
  topK = 5,
): Promise<StructuredReportResponse> {
  let res: Response;
  try {
    res = await fetch(apiUrl("/api/report/structured"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nc_path: ncPath, top_k: topK, format: "markdown" }),
    });
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    const msg = typeof d === "string" ? d : res.statusText;
    throw new Error(msg);
  }
  return res.json() as Promise<StructuredReportResponse>;
}

export function levelColor(level: string | undefined): string {
  const l = (level || "").toLowerCase();
  if (l === "high") return "rgba(239,68,68,0.22)";
  if (l === "medium") return "rgba(249,115,22,0.20)";
  if (l === "low") return "rgba(234,179,8,0.18)";
  return "transparent";
}

export function levelLabel(level: string | undefined): string {
  const l = (level || "").toLowerCase();
  if (l === "high") return "红色·高";
  if (l === "medium") return "橙色·中";
  if (l === "low") return "黄色·低";
  if (l === "normal") return "正常";
  return level || "—";
}
