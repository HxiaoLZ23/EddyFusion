const base = () => (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export async function uploadNcFiles(files: File[]): Promise<string[]> {
  const fd = new FormData();
  for (const f of files) {
    fd.append("files", f);
  }
  const res = await fetch(`${base()}/api/offline/nc`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || res.statusText);
  }
  const data = (await res.json()) as { paths: string[] };
  return data.paths;
}
