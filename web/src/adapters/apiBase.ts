/** FastAPI 根地址；开发环境未配置时走 Vite 代理的 `/api`。 */
export function apiBase(): string {
  return (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
}

export function apiUrl(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  const b = apiBase();
  return b ? `${b}${p}` : p;
}

export async function pingApiHealth(): Promise<boolean> {
  try {
    const res = await fetch(apiUrl("/api/health"), { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}

export function formatApiFetchError(err: unknown): string {
  const base = apiBase() || "（经 Vite 代理）http://localhost:5173/api";
  if (err instanceof TypeError) {
    return `无法连接后端 ${base}。请先在本机启动 API：在项目根执行 .\\scripts\\run_web_api.ps1`;
  }
  return err instanceof Error ? err.message : String(err);
}
