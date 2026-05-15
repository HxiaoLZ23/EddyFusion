const base = () => (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export type WindwaveOfflineReport = {
  status: string;
  report_text: string;
  anomaly_level?: string;
  anomaly_index?: number;
  typhoon_linked?: boolean;
  typhoon_link_note?: string;
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
