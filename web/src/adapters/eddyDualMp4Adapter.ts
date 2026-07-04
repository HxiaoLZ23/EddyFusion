const base = () => (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export type DualMp4Request = {
  nc_path: string;
  channel_mode?: "3ch";
  model_path?: string | null;
  conf?: number;
  iou?: number;
  base_imgsz?: number;
  fps?: number;
  max_frames?: number;
  time_stride?: number;
  time_start?: number;
  time_stop?: number | null;
  deliver?: "staged" | "full" | "base" | "annotate";
  job_id?: string | null;
};

export type DualMp4Response = {
  status: string;
  phase?: string;
  job_id?: string;
  preview_base?: string;
  preview_annotated?: string;
  fps: number;
  n_frames: number;
  time_labels: string[];
  truncated: boolean;
  time_indices: number[];
  video_encoding?: string;
  video_encoding_note?: string;
  detection_timeline?: {
    time?: string;
    peak_score?: number;
    max_conf?: number;
    mean_conf?: number;
    status?: string;
    count?: number;
  }[];
  meta?: Record<string, unknown>;
};

export function eddyPreviewUrl(filename: string): string {
  return `${base()}/api/eddy/preview?file=${encodeURIComponent(filename)}`;
}

export async function postEddyDualMp4(body: DualMp4Request): Promise<DualMp4Response> {
  const res = await fetch(`${base()}/api/eddy/dual-mp4`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ deliver: "full", ...body }),
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

export async function postEddyDualMp4Annotate(
  jobId: string,
  fps = 1,
  channelMode: "3ch" = "3ch",
): Promise<DualMp4Response> {
  const res = await fetch(`${base()}/api/eddy/dual-mp4/annotate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: jobId, fps, channel_mode: channelMode }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    const msg = typeof d === "string" ? d : res.statusText;
    throw new Error(msg);
  }
  return res.json() as Promise<DualMp4Response>;
}

/** 分阶段：先底图 MP4（无 YOLO），再批量 YOLO 出带框路；推理帧上限不变。 */
export async function postEddyDualMp4Staged(
  body: Omit<DualMp4Request, "deliver" | "job_id">,
): Promise<DualMp4Response> {
  const baseOut = await postEddyDualMp4({ ...body, deliver: "base" });
  if (!baseOut.job_id) {
    throw new Error("服务端未返回 job_id");
  }
  if (baseOut.preview_annotated) {
    return baseOut;
  }
  const annOut = await postEddyDualMp4Annotate(baseOut.job_id, body.fps ?? 1, body.channel_mode ?? "3ch");
  return {
    ...baseOut,
    preview_annotated: annOut.preview_annotated,
    phase: "complete",
    detection_timeline: annOut.detection_timeline ?? baseOut.detection_timeline,
  };
}
