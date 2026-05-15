import { uploadNcFiles } from "../adapters/ncOfflineAdapter";

export type DashboardMode = "offline" | "realtime";

type Props = {
  mode: DashboardMode;
  /** 离线：当前已选 NC（仓库相对路径） */
  offlineNcPath?: string | null;
  /** 离线：上传并校验通过后回调（单文件） */
  onOfflineNcUploaded?: (repoRelativePath: string) => void;
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
  onRtRefresh?: () => void | Promise<void>;
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
  uploadBusy,
  onUploadBusy,
  globalErr,
  onGlobalErr,
  rtStatus = "idle",
  rtFingerprint,
  rtError,
  rtIntervalSec = 30,
  onRtIntervalSec,
  onRtRefresh,
}: Props) {
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
      const ps = await uploadNcFiles([file]);
      const p = ps[0];
      if (p && onOfflineNcUploaded) {
        onOfflineNcUploaded(p);
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
          <label style={{ display: "inline-flex", alignItems: "center", gap: 8, fontSize: 13 }}>
            <span>上传 NC</span>
            <input type="file" accept=".nc,.nc4,.cdf" onChange={onFile} disabled={uploadBusy} />
          </label>
          {offlineNcPath ? (
            <span className="ocean-dashboard__badge" title={offlineNcPath}>
              已选：{shortNcLabel(offlineNcPath)}
            </span>
          ) : (
            <span style={{ fontSize: 12, color: "#64748b" }}>上传后自动跑涡旋 / 水文 / 风浪（无需再点）</span>
          )}
        </>
      ) : (
        <>
          <span className="ocean-dashboard__badge">实时 · 准 NC（占位）</span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 13 }}>
            连接器
            <span className="ocean-dashboard__status-dot" style={{ background: lamp, boxShadow: `0 0 6px ${lamp}` }} />
            {rtStatus === "ok" ? "就绪" : rtStatus === "err" ? "错误" : "未连接"}
          </span>
          <label style={{ fontSize: 13 }}>
            轮询间隔（秒）
            <input
              type="number"
              min={5}
              value={rtIntervalSec}
              onChange={(e) => onRtIntervalSec?.(Number(e.target.value))}
              style={{ width: 72, marginLeft: 6 }}
            />
          </label>
          <button type="button" onClick={() => void onRtRefresh?.()}>
            拉取 latest
          </button>
          {rtFingerprint && (
            <span style={{ fontSize: 11, color: "#64748b", maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis" }}>
              fp: {rtFingerprint}
            </span>
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
