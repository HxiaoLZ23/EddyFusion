import { apiUrl, formatApiFetchError } from "./apiBase";

export type EddyPreviewFrameRequest = {
  nc_path: string;
  time_index: number;
};

export type EddyPreviewStatRow = {
  id: number;
  area_px: number;
  perimeter_px?: number;
  bbox_xywh?: [number, number, number, number];
  centroid_xy: [number, number];
  /** YOLO 置信度，降级 ADT 模式下为 null */
  confidence?: number | null;
  class_id?: number | null;
  /** 气旋涡（冷涡）/ 反气旋涡（暖涡）；ADT 降级时为 null */
  eddy_type?: string | null;
  /** 轮廓顶点（像素坐标），供前端高亮叠加 */
  contour_xy?: [number, number][];
};

export type EddyPreviewFrameResponse = {
  status: string;
  /** "yolo" | "adt_fallback" */
  source?: string;
  time_index: number;
  time_label?: string | null;
  shape_hw: [number, number];
  image_data_url: string;
  stats_rows: EddyPreviewStatRow[];
  summary?: {
    candidate_count?: number;
    peak_conf?: number;
    /** 兼容旧版降级字段 */
    adt_threshold_p88?: number;
  };
};

export async function postEddyPreviewFrame(body: EddyPreviewFrameRequest): Promise<EddyPreviewFrameResponse> {
  let res: Response;
  try {
    res = await fetch(apiUrl("/api/eddy/preview-frame"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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
  return res.json() as Promise<EddyPreviewFrameResponse>;
}

