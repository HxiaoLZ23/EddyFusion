import { useCallback, useEffect, useRef, useState } from "react";
import { fetchLatestNc } from "../adapters/ncRealtimeAdapter";

export type RealtimeFeedState = {
  ncPaths: string[];
  fingerprint: string | null;
  status: "idle" | "ok" | "err";
  error: string | null;
  refresh: () => Promise<void>;
};

/** 准实时：轮询 latest，指纹变化时更新 ncPaths（单文件）。 */
export function useRealtimeNcFeed(enabled: boolean, intervalSec: number): RealtimeFeedState {
  const [ncPaths, setNcPaths] = useState<string[]>([]);
  const [fingerprint, setFingerprint] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "ok" | "err">("idle");
  const [error, setError] = useState<string | null>(null);
  const fpRef = useRef<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const j = await fetchLatestNc();
      setStatus("ok");
      setError(null);
      if (j.fingerprint !== fpRef.current) {
        fpRef.current = j.fingerprint;
        setFingerprint(j.fingerprint);
        setNcPaths([j.path]);
      }
    } catch (ex) {
      setStatus("err");
      setError(ex instanceof Error ? ex.message : String(ex));
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    void refresh();
  }, [enabled, refresh]);

  useEffect(() => {
    if (!enabled || intervalSec < 5) return;
    const id = window.setInterval(() => void refresh(), intervalSec * 1000);
    return () => window.clearInterval(id);
  }, [enabled, intervalSec, refresh]);

  return { ncPaths, fingerprint, status, error, refresh };
}
