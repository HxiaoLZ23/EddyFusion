const base = () => (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export type DualMp4Request = {
  nc_path: string;
  model_path?: string | null;
  conf?: number;
  iou?: number;
  base_imgsz?: number;
  fps?: number;
  max_frames?: number;
  time_stride?: number;
  time_start?: number;
  time_stop?: number | null;
};

export type DualMp4Response = {
  status: string;
  preview_base: string;
  preview_annotated: string;
  fps: number;
  n_frames: number;
  time_labels: string[];
  truncated: boolean;
  time_indices: number[];
  video_encoding?: string;
  video_encoding_note?: string;
  meta?: Record<string, unknown>;
};

export function eddyPreviewUrl(filename: string): string {
  return `${base()}/api/eddy/preview?file=${encodeURIComponent(filename)}`;
}

export async function postEddyDualMp4(body: DualMp4Request): Promise<DualMp4Response> {
  const res = await fetch(`${base()}/api/eddy/dual-mp4`, {
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
  return res.json() as Promise<DualMp4Response>;
}
