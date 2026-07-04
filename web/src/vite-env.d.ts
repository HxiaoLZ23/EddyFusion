/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE: string;
  readonly VITE_TYPHOON_KB_URL?: string;
  /** 设为 `true` 时展示水文大屏区块与 /l1/hydro；默认关闭 */
  readonly VITE_SHOW_HYDRO?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
