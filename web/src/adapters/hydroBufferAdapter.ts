const base = () => (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export type HydroMetaResponse = {
  T_need: number;
  T_hat: number;
  buffer_sufficient: boolean;
};

export async function postHydroMeta(
  ncPaths: string[],
  configPath = "config/hydro_hycom_l2.yaml",
): Promise<HydroMetaResponse> {
  const res = await fetch(`${base()}/api/hydro/meta`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nc_paths: ncPaths, config_path: configPath }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || res.statusText);
  }
  return res.json() as Promise<HydroMetaResponse>;
}
