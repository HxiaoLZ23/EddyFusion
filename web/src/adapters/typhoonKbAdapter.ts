import { apiUrl, formatApiFetchError } from "./apiBase";

export type TyphoonKbStatus = {
  ready: boolean;
  events_json_path: string;
  events_count: number;
  source?: string | null;
  seed_hint?: string;
  full_build_hint?: string;
};

export type TyphoonKbDefaults = {
  start_time: string;
  end_time: string;
  lon_min: number;
  lon_max: number;
  lat_min: number;
  lat_max: number;
  top_k: number;
  events_json_path: string;
  demo_cases_path: string;
};

export type TyphoonCandidate = {
  event_id?: string;
  name?: string;
  start_time?: string;
  end_time?: string;
  intensity_level?: string;
  peak_wind_kt?: number;
  bbox_overlap_ratio?: number;
  time_overlap_hours?: number;
  score?: number;
  dtw_distance?: number;
  summary?: string;
};

export type TyphoonQueryResult = {
  status: string;
  count: number;
  query: Record<string, unknown>;
  candidates: TyphoonCandidate[];
  events_json_path: string;
};

export type TyphoonEventRow = {
  event_id?: string;
  name?: string;
  season?: string;
  intensity_level?: string;
  start_time?: string;
  end_time?: string;
  center_lon?: number;
  center_lat?: number;
  peak_wind_kt?: number;
  lon_min?: number;
  lon_max?: number;
  lat_min?: number;
  lat_max?: number;
  n_points?: number;
  retrieval_keys?: string[];
};

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    const msg =
      typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : res.statusText;
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export async function fetchTyphoonKbStatus(): Promise<TyphoonKbStatus> {
  try {
    const res = await fetch(apiUrl("/api/typhoon-kb/status"));
    return parseJson(res);
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
}

export async function fetchTyphoonKbDefaults(): Promise<TyphoonKbDefaults> {
  try {
    const res = await fetch(apiUrl("/api/typhoon-kb/defaults"));
    return parseJson(res);
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
}

export async function postTyphoonKbQuery(body: {
  start_time: string;
  end_time: string;
  lon_min: number;
  lon_max: number;
  lat_min: number;
  lat_max: number;
  top_k: number;
  events_json_path?: string;
}): Promise<TyphoonQueryResult> {
  try {
    const res = await fetch(apiUrl("/api/typhoon-kb/query"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return parseJson(res);
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
}

export async function fetchTyphoonKbEvents(params: {
  page?: number;
  page_size?: number;
  keyword?: string;
  level?: string;
  season?: string;
  events_json_path?: string;
}): Promise<{
  total: number;
  page: number;
  page_size: number;
  max_page: number;
  items: TyphoonEventRow[];
  facets: { levels: string[]; seasons: string[] };
}> {
  const q = new URLSearchParams();
  if (params.page != null) q.set("page", String(params.page));
  if (params.page_size != null) q.set("page_size", String(params.page_size));
  if (params.keyword) q.set("keyword", params.keyword);
  if (params.level) q.set("level", params.level);
  if (params.season) q.set("season", params.season);
  if (params.events_json_path) q.set("events_json_path", params.events_json_path);
  try {
    const res = await fetch(apiUrl(`/api/typhoon-kb/events?${q.toString()}`));
    return parseJson(res);
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
}

export async function fetchTyphoonDemoCases(path?: string): Promise<{ cases: unknown[]; path: string; note?: string }> {
  const q = path ? `?path=${encodeURIComponent(path)}` : "";
  try {
    const res = await fetch(apiUrl(`/api/typhoon-kb/demo-cases${q}`));
    return parseJson(res);
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
}

export function friendlyTyphoonLevel(level: string | undefined): string {
  const m: Record<string, string> = {
    typhoon: "台风",
    tropical_storm: "热带风暴",
    tropical_depression: "热带低压",
    unknown: "未知",
  };
  return m[level || ""] ?? level ?? "—";
}
