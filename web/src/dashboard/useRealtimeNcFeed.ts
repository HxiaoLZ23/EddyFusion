import { useCallback, useEffect, useRef, useState } from "react";
import { fetchLatestNc, fetchRealtimeStatus, type RealtimeStatus } from "../adapters/ncRealtimeAdapter";

export type RealtimeFeedState = {
  ncPaths: string[];
  fingerprint: string | null;
  status: "idle" | "ok" | "err";
  error: string | null;
  connector: RealtimeStatus | null;
  newFileDetected: boolean;
  refresh: () => Promise<void>;
  refreshStatus: () => Promise<void>;
};

/**
 * G15 准实时：轮询 /realtime/status + /latest；仅在 armed 后自动分析。
 */
export function useRealtimeNcFeed(enabled: boolean, intervalSec: number, armed: boolean): RealtimeFeedState {
  const [ncPaths, setNcPaths] = useState<string[]>([]);
  const [fingerprint, setFingerprint] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "ok" | "err">("idle");
  const [error, setError] = useState<string | null>(null);
  const [connector, setConnector] = useState<RealtimeStatus | null>(null);
  const [newFileDetected, setNewFileDetected] = useState(false);
  const fpRef = useRef<string | null>(null);

  useEffect(() => {
    if (armed) return;
    fpRef.current = null;
    setNcPaths([]);
    setFingerprint(null);
    setStatus("idle");
    setError(null);
    setNewFileDetected(false);
  }, [armed]);

  const refreshStatus = useCallback(async () => {
    try {
      const st = await fetchRealtimeStatus();
      setConnector(st);
      if (!st.connected) {
        setStatus("err");
        setError(st.error || "连接器未就绪");
      }
    } catch (ex) {
      setConnector(null);
    }
  }, []);

  const refresh = useCallback(async () => {
    await refreshStatus();
    try {
      const j = await fetchLatestNc();
      setStatus("ok");
      setError(null);
      const changed = fpRef.current !== null && j.fingerprint !== fpRef.current;
      if (j.fingerprint !== fpRef.current) {
        fpRef.current = j.fingerprint;
        setFingerprint(j.fingerprint);
        setNcPaths([j.path]);
        if (changed) setNewFileDetected(true);
      }
    } catch (ex) {
      setStatus("err");
      setError(ex instanceof Error ? ex.message : String(ex));
    }
  }, [refreshStatus]);

  useEffect(() => {
    if (!enabled) return;
    void refreshStatus();
  }, [enabled, refreshStatus]);

  useEffect(() => {
    if (!enabled || !armed) return;
    void refresh();
  }, [enabled, armed, refresh]);

  useEffect(() => {
    if (!enabled || !armed || intervalSec < 5) return;
    const id = window.setInterval(() => void refresh(), intervalSec * 1000);
    return () => window.clearInterval(id);
  }, [enabled, armed, intervalSec, refresh]);

  return { ncPaths, fingerprint, status, error, connector, newFileDetected, refresh, refreshStatus };
}
