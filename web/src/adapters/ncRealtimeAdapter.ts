import { apiUrl, formatApiFetchError } from "./apiBase";

export type LatestNc = {
  path: string;
  filename?: string;
  mtime: string;
  mtime_iso?: string;
  fingerprint: string;
  size_bytes?: number;
  size_mb?: number;
  age_sec?: number;
  stale?: boolean;
};

export type RealtimeStatus = {
  connected: boolean;
  ready: boolean;
  source?: string;
  poll_dir?: string;
  poll_interval_hint_sec?: number;
  stale_threshold_sec?: number;
  nc_count?: number;
  latest?: LatestNc | null;
  checked_at_iso?: string;
  error?: string;
};

export async function fetchRealtimeStatus(): Promise<RealtimeStatus> {
  let res: Response;
  try {
    res = await fetch(apiUrl("/api/realtime/status"));
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || res.statusText);
  }
  return res.json() as Promise<RealtimeStatus>;
}

export async function fetchLatestNc(): Promise<LatestNc> {
  const res = await fetch(apiUrl("/api/realtime/latest"));
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || res.statusText);
  }
  return res.json() as Promise<LatestNc>;
}
