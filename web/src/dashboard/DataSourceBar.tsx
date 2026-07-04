import { useEffect, useState } from "react";
import { apiBase, pingApiHealth } from "../adapters/apiBase";
import { uploadNcFiles } from "../adapters/ncOfflineAdapter";
import type { RealtimeStatus } from "../adapters/ncRealtimeAdapter";
export type DashboardMode = "offline" | "realtime";

type Props = {
  mode: DashboardMode;
  /** 离线：当前已选 NC（仓库相对路径） */
  offlineNcPath?: string | null;
  /** 离线：上传并校验通过后回调（单文件） */
  onOfflineNcUploaded?: (repoRelativePath: string) => void;
  /** 监测总览左栏 TaskConfigPanel 已含上传时隐藏此处上传控件 */
  hideOfflineUpload?: boolean;
  uploadBusy: boolean;
  onUploadBusy: (v: boolean) => void;
  globalErr: string | null;
  onGlobalErr: (msg: string | null) => void;
  /** 实时模式 */
  rtStatus?: "idle" | "ok" | "err";
  rtFingerprint?: string | null;
  rtError?: string | null;
  rtIntervalSec?: number;
  onRtIntervalSec?: (n: number) => void;
  /** 已点击拉取并允许轮询 */
  rtArmed?: boolean;
  onRtArmAndRefresh?: () => void | Promise<void>;
  rtConnector?: RealtimeStatus | null;
  rtNewFile?: boolean;
};

function shortNcLabel(path: string): string {
  const p = path.replace(/\\/g, "/");
  const i = p.lastIndexOf("/");
  return i >= 0 ? p.slice(i + 1) : p;
}

export function DataSourceBar({
  mode,
  offlineNcPath = null,
  onOfflineNcUploaded,
  hideOfflineUpload = false,
  uploadBusy,
  onUploadBusy,
  globalErr,
  onGlobalErr,
  rtStatus = "idle",
  rtFingerprint,
  rtError,
  rtIntervalSec = 30,
  onRtIntervalSec,
  rtArmed = false,
  onRtArmAndRefresh,
  rtConnector = null,
  rtNewFile = false,
}: Props) {
  const [apiReachable, setApiReachable] = useState<boolean | null>(null);

  useEffect(() => {
    if (mode !== "offline") return;
    let cancelled = false;
    void pingApiHealth().then((ok) => {
      if (!cancelled) setApiReachable(ok);
    });
    return () => {
      cancelled = true;
    };
  }, [mode]);

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const okName = /\.(nc|nc4|cdf)$/i.test(file.name);
    if (!okName) {
      onGlobalErr("请上传 NetCDF：.nc / .nc4 / .cdf");
      e.target.value = "";
      return;
    }
    onGlobalErr(null);
    onUploadBusy(true);
    try {
      const ok = await pingApiHealth();
      setApiReachable(ok);
      if (!ok) {
        throw new Error(
          `后端 API 未响应（${apiBase() || "请配置 VITE_API_BASE 或使用 Vite 代理"}）。请先运行 scripts/run_web_api.ps1`,
        );
      }
      const ps = await uploadNcFiles([file]);
      const p = ps[0];
      if (p && onOfflineNcUploaded) {
        onOfflineNcUploaded(p);
        setApiReachable(true);
      }
    } catch (ex) {
      onGlobalErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      onUploadBusy(false);
      e.target.value = "";
    }
  };

  const lamp =
    rtStatus === "ok" ? "#16a34a" : rtStatus === "err" ? "#dc2626" : "#94a3b8";

  return (
    <div className="ocean-dashboard__compact">
      {mode === "offline" ? (
        <>
          <span className="ocean-dashboard__badge">离线 · 单文件 NC</span>
          <span className="ocean-dashboard__badge" style={{ background: "#f0fdf4", color: "#166534" }}>
            涡旋 3ch（ADT+ADT+ADT）
          </span>
          {!hideOfflineUpload && (
            <label style={{ display: "inline-flex", alignItems: "center", gap: 8, fontSize: 13 }}>
              <span>上传 NC</span>
              <input type="file" accept=".nc,.nc4,.cdf" onChange={onFile} disabled={uploadBusy} />
            </label>
          )}
          {uploadBusy && (
            <span className="ocean-dashboard__badge" style={{ background: "#e0f2fe", color: "#0369a1" }}>
              上传校验中…
            </span>
          )}
          {apiReachable === false && !uploadBusy && (
            <span style={{ fontSize: 12, color: "#b91c1c", maxWidth: 420 }}>
              后端未连接 · 请启动 <code style={{ fontSize: 11 }}>scripts/run_web_api.ps1</code>
            </span>
          )}
          {offlineNcPath ? (
            <span className="ocean-dashboard__badge" title={offlineNcPath}>
              已选：{shortNcLabel(offlineNcPath)}
            </span>
          ) : (
            <span style={{ fontSize: 12, color: "#64748b" }}>
              上传后自动跑涡旋与风浪分析
            </span>
          )}
        </>
      ) : (
        <>
          <span className="ocean-dashboard__badge">实时 · 目录轮询</span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 13 }}>
            连接器
            <span className="ocean-dashboard__status-dot" style={{ background: lamp, boxShadow: `0 0 6px ${lamp}` }} />
            {!rtArmed
              ? "待拉取"
              : rtStatus === "ok"
                ? "就绪"
                : rtStatus === "err"
                  ? "错误"
                  : "连接中"}
          </span>
          <label style={{ fontSize: 13 }}>
            轮询间隔（秒）
            <input
              type="number"
              min={5}
              value={rtIntervalSec}
              onChange={(e) => onRtIntervalSec?.(Number(e.target.value))}
              style={{ width: 72, marginLeft: 6 }}
              disabled={!rtArmed}
            />
          </label>
          <button type="button" onClick={() => void onRtArmAndRefresh?.()}>
            {rtArmed ? "再次拉取 latest" : "拉取 latest"}
          </button>
          {rtArmed && rtFingerprint && (
            <span style={{ fontSize: 11, color: "#64748b", maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis" }}>
              fp: {rtFingerprint}
            </span>
          )}
          {!rtArmed && (
            <span style={{ fontSize: 12, color: "#64748b" }}>
              进入本页不会自动分析；需手动拉取 latest 后才跑涡旋与风浪
            </span>
          )}
          {rtConnector?.connected && (
            <span style={{ fontSize: 11, color: "#64748b" }} title={rtConnector.poll_dir}>
              目录 {rtConnector.nc_count ?? 0} 个 NC
              {rtConnector.latest?.mtime_iso ? ` · 最新 ${rtConnector.latest.mtime_iso}` : ""}
              {rtConnector.latest?.stale ? " · 已陈旧" : ""}
            </span>
          )}
          {rtNewFile && (
            <span className="ocean-dashboard__badge" style={{ background: "#fef3c7", color: "#92400e" }}>
              检测到新文件
            </span>
          )}
          {rtArmed && rtStatus === "ok" && (
            <span style={{ fontSize: 12, color: "#64748b" }}>已连接，按间隔轮询；结果仅保存在实时会话</span>
          )}
          {rtError && <span style={{ fontSize: 12, color: "#b91c1c" }}>{rtError}</span>}
        </>
      )}
      {globalErr && mode === "offline" && (
        <span style={{ fontSize: 12, color: "#b91c1c" }}>{globalErr}</span>
      )}
    </div>
  );
}
