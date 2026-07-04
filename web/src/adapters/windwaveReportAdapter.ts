const base = () => (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export type WindwaveOfflineReport = {
  status: string;
  report_text: string;
  anomaly_level?: string;
  anomaly_index?: number;
  wind_wave_series?: WindWaveSeriesPoint[];
  typhoon_linked?: boolean;
  typhoon_link_note?: string;
  typhoon_candidates?: Record<string, unknown>[];
  typhoon_events_path?: string;
  typhoon_retrieval?: Record<string, unknown>;
  typhoon_query?: {
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
};

export type WindWaveSeriesPoint = {
  step: number;
  wind_observed: number;
  wind_predicted: number;
  wave_observed: number;
  wave_predicted: number;
};

export async function postWindwaveOfflineReport(ncPath: string): Promise<WindwaveOfflineReport> {
  const res = await fetch(`${base()}/api/windwave/offline-report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nc_path: ncPath }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    const msg =
      typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : res.statusText;
    throw new Error(msg);
  }
  return res.json() as Promise<WindwaveOfflineReport>;
}
