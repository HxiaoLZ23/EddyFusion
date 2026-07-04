import { apiUrl, formatApiFetchError } from "./apiBase";

export type ReportListItem = {
  id: string;
  created_at?: number;
  created_at_iso?: string;
  nc_path?: string;
  source?: string;
  mode?: string;
  title?: string;
  anomaly_level?: string;
  anomaly_index?: number;
};

export type SavedReport = ReportListItem & {
  markdown: string;
  fields?: Record<string, unknown>;
};

export async function fetchReportHistory(limit = 50): Promise<ReportListItem[]> {
  let res: Response;
  try {
    res = await fetch(apiUrl(`/api/report/history?limit=${limit}`));
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    throw new Error(typeof d === "string" ? d : res.statusText);
  }
  const data = (await res.json()) as { reports?: ReportListItem[] };
  return data.reports ?? [];
}

export async function fetchReportById(id: string): Promise<SavedReport> {
  let res: Response;
  try {
    res = await fetch(apiUrl(`/api/report/${encodeURIComponent(id)}`));
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    throw new Error(typeof d === "string" ? d : res.statusText);
  }
  const data = (await res.json()) as { report: SavedReport };
  return data.report;
}

export async function postSaveReport(body: {
  nc_path: string;
  markdown: string;
  fields?: Record<string, unknown>;
  source?: "windwave" | "eddy" | "combined";
  mode?: "offline" | "realtime";
  title?: string;
}): Promise<{ id: string }> {
  let res: Response;
  try {
    res = await fetch(apiUrl("/api/report/save"), {
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
    throw new Error(typeof d === "string" ? d : res.statusText);
  }
  const data = (await res.json()) as { id: string };
  return { id: data.id };
}

export function downloadMarkdown(filename: string, content: string): void {
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename.endsWith(".md") ? filename : `${filename}.md`;
  a.click();
  URL.revokeObjectURL(url);
}
