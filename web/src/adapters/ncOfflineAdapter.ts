import { apiUrl, formatApiFetchError } from "./apiBase";

export async function uploadNcFiles(files: File[]): Promise<string[]> {
  const fd = new FormData();
  for (const f of files) {
    fd.append("files", f);
  }
  let res: Response;
  try {
    res = await fetch(apiUrl("/api/offline/nc"), { method: "POST", body: fd });
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    const msg =
      typeof d === "string" ? d : Array.isArray(d) ? JSON.stringify(d) : res.statusText;
    throw new Error(msg || `上传失败 HTTP ${res.status}`);
  }
  const data = (await res.json()) as { paths: string[] };
  if (!data.paths?.length) {
    throw new Error("服务端未返回 NC 路径");
  }
  return data.paths;
}
