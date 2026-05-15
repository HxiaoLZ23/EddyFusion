const base = () => (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export type LatestNc = { path: string; mtime: string; fingerprint: string };

export async function fetchLatestNc(): Promise<LatestNc> {
  const res = await fetch(`${base()}/api/realtime/latest`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || res.statusText);
  }
  return res.json() as Promise<LatestNc>;
}
