import { useMemo, useState } from "react";
import "./ocean-dashboard.css";
import { DataSourceBar } from "./DataSourceBar";
import { EddyPanel } from "./EddyPanel";
import { HydroPanel } from "./HydroPanel";
import { WindWavePanel } from "./WindWavePanel";
import { useRealtimeNcFeed } from "./useRealtimeNcFeed";

export type OceanDashboardProps = {
  mode: "offline" | "realtime";
};

export function OceanDashboard({ mode }: OceanDashboardProps) {
  const [offlineNc, setOfflineNc] = useState<string | null>(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [globalErr, setGlobalErr] = useState<string | null>(null);
  const [rtInterval, setRtInterval] = useState(30);

  const rt = useRealtimeNcFeed(mode === "realtime", rtInterval);

  const ncPaths = useMemo(() => {
    if (mode === "offline") {
      return offlineNc ? [offlineNc] : [];
    }
    return rt.ncPaths;
  }, [mode, offlineNc, rt.ncPaths]);

  const title = mode === "offline" ? "离线系统" : "实时系统";
  const caption =
    mode === "offline"
      ? "单文件上传 → 校验 NetCDF → 自动：涡旋双路视频、水文曲线与热力图、风浪规则报告（与规划 §2 同屏）。"
      : "与离线共用同屏布局；数据源为准实时 NC（latest 占位），涡旋需手动点「生成双路视频」。";

  const eddyNcPath = mode === "offline" ? (offlineNc ?? "") : (ncPaths[0] ?? "");

  return (
    <div className="ocean-dashboard">
      <div style={{ marginBottom: 10 }}>
        <h1 className="ocean-dashboard__title">{title}</h1>
        <p className="ocean-dashboard__caption">{caption}</p>
      </div>

      <DataSourceBar
        mode={mode}
        offlineNcPath={mode === "offline" ? offlineNc : null}
        onOfflineNcUploaded={mode === "offline" ? (p) => setOfflineNc(p) : undefined}
        uploadBusy={uploadBusy}
        onUploadBusy={setUploadBusy}
        globalErr={globalErr}
        onGlobalErr={setGlobalErr}
        rtStatus={rt.status}
        rtFingerprint={rt.fingerprint}
        rtError={rt.error}
        rtIntervalSec={rtInterval}
        onRtIntervalSec={setRtInterval}
        onRtRefresh={rt.refresh}
      />

      <div className="ocean-dashboard__main">
        <aside className="ocean-dashboard__eddy">
          <EddyPanel ncPath={eddyNcPath} autoGenerate={mode === "offline"} />
        </aside>
        <div className="ocean-dashboard__right">
          <HydroPanel ncPaths={ncPaths} autoLoadOnPathChange />
          <WindWavePanel ncPath={mode === "offline" ? offlineNc : ncPaths[0] ?? null} autoRun={mode === "offline"} />
        </div>
      </div>
    </div>
  );
}
