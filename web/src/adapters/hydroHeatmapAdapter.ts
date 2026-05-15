const base = () => (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export type HeatmapPayload = {
  nc_paths: string[];
  config_path?: string;
  ckpt_path?: string;
  sample_index?: number;
  window_stride?: number;
  max_windows?: number;
  feature?: string;
  kind?: string;
  lead_time_index?: number;
};

export type HeatmapResponse = {
  lons: number[];
  lats: number[];
  values: (number | null)[][];
  feature: string;
  kind: string;
  lead_time_index: number;
  crs: string;
  warnings: string[];
  feature_names: string[];
  curve_data: Record<string, { horizon: number; gt: number; pred: number }[]>;
  inference: Record<string, unknown>;
  meta: {
    T_need: number;
    T_hat: number;
    buffer_sufficient: boolean;
    materialize?: Record<string, unknown>;
  };
};

export async function postHydroHeatmap(body: HeatmapPayload): Promise<HeatmapResponse> {
  const res = await fetch(`${base()}/api/hydro/heatmap`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    const msg =
      typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : res.statusText;
    throw new Error(msg);
  }
  return res.json() as Promise<HeatmapResponse>;
}
