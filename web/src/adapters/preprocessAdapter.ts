import { apiUrl, formatApiFetchError } from "./apiBase";

export type NcMetaResponse = {
  nc_path: string;
  source_path?: string;
  variables?: string[];
  dimensions?: Record<string, number>;
  time_len?: number;
  time_start_label?: string;
  time_end_label?: string;
  lat_min?: number;
  lat_max?: number;
  lon_min?: number;
  lon_max?: number;
  variable_map?: {
    found: Record<string, string>;
    eddy_ready?: boolean;
    windwave_ready?: boolean;
  };
};

export type SubsetRequest = {
  nc_path: string;
  time_start?: number | null;
  time_stop?: number | null;
  lon_min?: number | null;
  lon_max?: number | null;
  lat_min?: number | null;
  lat_max?: number | null;
  task?: "eddy" | "windwave" | null;
};

export type SubsetResponse = {
  status: string;
  nc_path: string;
  source_nc_path?: string;
  task?: string | null;
  applied?: {
    time?: { start_index: number; stop_index: number; dim: string } | null;
    bbox?: Record<string, unknown> | null;
  };
  dimensions?: Record<string, number>;
  variable_map?: NcMetaResponse["variable_map"];
  size_mb?: number;
};

async function parseJsonError(res: Response): Promise<string> {
  const err = await res.json().catch(() => ({}));
  const d = (err as { detail?: unknown }).detail;
  return typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : res.statusText;
}

export async function fetchNcMeta(ncPath: string): Promise<NcMetaResponse> {
  const q = new URLSearchParams({ nc_path: ncPath });
  let res: Response;
  try {
    res = await fetch(apiUrl(`/api/preprocess/meta?${q}`));
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
  if (!res.ok) throw new Error(await parseJsonError(res));
  return res.json() as Promise<NcMetaResponse>;
}

export async function postNcSubset(body: SubsetRequest): Promise<SubsetResponse> {
  let res: Response;
  try {
    res = await fetch(apiUrl("/api/preprocess/subset"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
  if (!res.ok) throw new Error(await parseJsonError(res));
  return res.json() as Promise<SubsetResponse>;
}
