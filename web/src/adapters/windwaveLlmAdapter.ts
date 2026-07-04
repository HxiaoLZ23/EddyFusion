const base = () => (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export type LlmReportResponse = {
  status: string;
  summary_anomaly?: string;
  impact?: string;
  historical_analogy?: string;
  actions?: string[];
  fingerprint?: string;
};

export async function postWindwaveLlmReport(
  ncPath: string,
  model?: string,
): Promise<LlmReportResponse> {
  const res = await fetch(`${base()}/api/windwave/llm-report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nc_path: ncPath, model: model || null }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    const msg = typeof d === "string" ? d : res.statusText;
    throw new Error(msg);
  }
  return res.json() as Promise<LlmReportResponse>;
}
