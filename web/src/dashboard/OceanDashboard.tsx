import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./ocean-dashboard.css";
import { CollapsedStrip } from "./CollapsedStrip";
import { DataSourceBar } from "./DataSourceBar";
import { EddyHistoryDrawer } from "./EddyHistoryDrawer";
import { EddyPanel } from "./EddyPanel";
import { HydroLevel1View } from "./HydroLevel1View";
import { HydroPanel } from "./HydroPanel";
import { type OceanMode, useDashboardModeLifecycle, useOceanSession } from "./offlineSession";
import { PanelLevel1Chrome } from "./PanelLevel1Chrome";
import { SHOW_HYDRO_UI } from "../featureFlags";
import { useRealtimeNcFeed } from "./useRealtimeNcFeed";
import { TyphoonKbPanel } from "./TyphoonKbPanel";
import { WindWaveLevel1View } from "./WindWaveLevel1View";
import { WindWavePanel } from "./WindWavePanel";

export type OceanDashboardProps = {
  mode: OceanMode;
  /** 左栏 TaskConfigPanel 已负责离线上传 */
  hideOfflineUploadInBar?: boolean;
  /** 监测总览页是否处于当前可见路由（隐藏时暂停准实时轮询） */
  active?: boolean;
};

const L1_PANELS = new Set<string>(
  SHOW_HYDRO_UI ? ["eddy", "hydro", "windwave"] : ["eddy", "windwave"],
);
type L1Panel = "eddy" | "hydro" | "windwave";

function parseL1Panel(pathname: string, mode: OceanMode): L1Panel | undefined {
  const prefix = mode === "offline" ? "/offline" : "/realtime";
  const inner = SHOW_HYDRO_UI ? "eddy|hydro|windwave" : "eddy|windwave";
  const m = pathname.match(new RegExp(`^${prefix}/l1/(${inner})/?$`));
  if (!m || !L1_PANELS.has(m[1])) return undefined;
  return m[1] as L1Panel;
}

function modeBase(mode: OceanMode): string {
  return mode === "offline" ? "/offline" : "/realtime";
}

export function OceanDashboard({ mode, hideOfflineUploadInBar = false, active = true }: OceanDashboardProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const panel = parseL1Panel(location.pathname, mode);
  const session = useOceanSession(mode);
  const pipelineArmed = session.pipelineArmed;
  useDashboardModeLifecycle(mode);

  const [uploadBusy, setUploadBusy] = useState(false);
  const [globalErr, setGlobalErr] = useState<string | null>(null);
  const [rtInterval, setRtInterval] = useState(30);
  /** 实时：须用户点击「拉取 latest」后才连接并跑三模块（不与离线上传自动联动） */
  const [rtArmed, setRtArmed] = useState(false);

  useEffect(() => {
    if (mode === "realtime") setRtArmed(false);
  }, [mode]);

  const rt = useRealtimeNcFeed(mode === "realtime" && active, rtInterval, rtArmed);

  const ncPaths = useMemo(() => {
    if (mode === "offline") {
      return session.ncPath ? [session.ncPath] : [];
    }
    return rt.ncPaths;
  }, [mode, session.ncPath, rt.ncPaths]);

  const hydroNcPaths = useMemo(() => {
    if (session.hydroBufferPaths.length > 0) {
      return session.hydroBufferPaths;
    }
    return ncPaths;
  }, [session.hydroBufferPaths, ncPaths]);

  useEffect(() => {
    if (mode !== "realtime" || !rtArmed || !ncPaths[0]) return;
    session.setNcPath(ncPaths[0]);
    if (SHOW_HYDRO_UI) session.appendHydroBuffer(ncPaths[0]);
  }, [mode, rtArmed, ncPaths.join("|")]);

  const title = mode === "offline" ? "监测总览 · 离线 NC" : "监测总览 · 准实时";
  const caption =
    mode === "offline"
      ? "上传 NetCDF → 自动：3ch 涡旋双路视频 + 风浪规则报告。顶栏可进入涡旋/风浪/报告专页（Phase 2+ 左-中-右布局）。"
      : "点击「拉取 latest」后自动跑 3ch 涡旋与风浪（与离线会话隔离）。";

  const eddyNcPath = ncPaths[0] ?? "";
  const windNcPath = mode === "offline" ? session.ncPath : ncPaths[0] ?? null;
  const hasNc = !!eddyNcPath;
  const pipelineActive = mode === "offline" ? hasNc && pipelineArmed : hasNc && rtArmed && pipelineArmed;

  useEffect(() => {
    const base = modeBase(mode);
    const m = location.pathname.match(new RegExp(`^${base}/l1/([^/]+)`));
    if (m && !L1_PANELS.has(m[1])) {
      navigate(base, { replace: true });
    }
  }, [mode, location.pathname, navigate]);

  useEffect(() => {
    if (SHOW_HYDRO_UI) return;
    const base = modeBase(mode);
    if (location.pathname.startsWith(`${base}/l1/hydro`)) {
      navigate(base, { replace: true });
    }
  }, [SHOW_HYDRO_UI, mode, location.pathname, navigate]);

  const goL1 = (p: L1Panel) => {
    navigate(`${modeBase(mode)}/l1/${p}`);
  };
  const goHome = () => navigate(modeBase(mode));
  const exportSessionSummary = () => {
    const payload = {
      mode,
      nc_path: session.ncPath,
      eddy_channel_mode: session.eddyChannelMode,
      windwave_level: session.windwave.anomalyLevel ?? null,
      windwave_index: session.windwave.anomalyIndex ?? null,
      typhoon_candidates: session.windwave.typhoonCandidates ?? [],
      eddy_jobs: session.eddyHistory,
      exported_at: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `eddyfusion_${mode}_summary.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const parkHome = !!panel && panel !== "eddy";

  const homeLayerClass = [
    "ocean-dashboard__home-layer",
    parkHome ? "ocean-dashboard__home-layer--parked" : "",
    panel === "eddy" ? "ocean-dashboard__home-layer--l1-eddy" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const homeMain = (
    <div className="ocean-dashboard__main" aria-hidden={parkHome}>
      <aside className="ocean-dashboard__eddy">
        {panel === "eddy" && (
          <div className="ocean-dashboard__l1-back-row ocean-dashboard__l1-back-row--overlay">
            <button type="button" className="ocean-dashboard__l1-back" onClick={goHome}>
              ← 返回同屏
            </button>
          </div>
        )}
        <div className="ocean-dashboard__eddy-inner">
          <PanelLevel1Chrome>
            <EddyPanel mode={mode} ncPath={eddyNcPath} autoGenerate={pipelineActive} />
          </PanelLevel1Chrome>
          {panel === "eddy" && (
            <EddyHistoryDrawer
              rows={session.eddyHistory}
              detectionFrames={session.eddyDetectionFrames}
              onClose={goHome}
            />
          )}
        </div>
      </aside>
      <div className="ocean-dashboard__right">
        {SHOW_HYDRO_UI ? (
          <PanelLevel1Chrome>
            <HydroPanel
              mode={mode}
              ncPaths={hydroNcPaths}
              autoLoadOnPathChange={pipelineActive && hydroNcPaths.length > 0}
            />
          </PanelLevel1Chrome>
        ) : (
          <TyphoonKbPanel mode={mode} />
        )}
        <PanelLevel1Chrome>
          <WindWavePanel mode={mode} ncPath={windNcPath} autoRun={pipelineActive} />
        </PanelLevel1Chrome>
      </div>
    </div>
  );

  const l1Main =
    panel && panel !== "eddy" ? (
      <div className={`ocean-dashboard__main ocean-dashboard__main--l1 ocean-dashboard__main--l1-${panel}`}>
        {panel === "hydro" && SHOW_HYDRO_UI ? (
          <div className="ocean-dashboard__l1-strip-col">
            <CollapsedStrip title="涡旋" accent="sky" hint="进入涡旋二级" onClick={() => goL1("eddy")} />
          </div>
        ) : (
          <div className="ocean-dashboard__l1-strip-col">
            <CollapsedStrip title="涡旋" accent="sky" hint="进入涡旋二级" onClick={() => goL1("eddy")} />
            {SHOW_HYDRO_UI ? (
              <CollapsedStrip title="水文" accent="sky" hint="进入水文二级" onClick={() => goL1("hydro")} />
            ) : null}
          </div>
        )}

        <div className={`ocean-dashboard__right ocean-dashboard__right--l1 ocean-dashboard__right--l1-${panel}`}>
          {panel === "hydro" && SHOW_HYDRO_UI ? (
            <>
              <div className="ocean-dashboard__panel ocean-dashboard__panel--hydro ocean-dashboard__panel--l1-expanded">
                <div className="ocean-dashboard__l1-back-row">
                  <button type="button" className="ocean-dashboard__l1-back" onClick={goHome}>
                    ← 返回同屏
                  </button>
                </div>
                <HydroLevel1View
                  mode={mode}
                  curveData={session.hydro.curveData}
                  featureNames={session.hydro.featureNames}
                />
              </div>
              <CollapsedStrip title="风浪" accent="amber" layout="bar" onClick={() => goL1("windwave")} />
            </>
          ) : (
            <div className="ocean-dashboard__panel ocean-dashboard__panel--windwave ocean-dashboard__panel--l1-expanded">
              <div className="ocean-dashboard__l1-back-row">
                <button type="button" className="ocean-dashboard__l1-back" onClick={goHome}>
                  ← 返回同屏
                </button>
              </div>
              <WindWaveLevel1View
                mode={mode}
                reportText={session.windwave.reportText}
                anomalyLevel={session.windwave.anomalyLevel}
                typhoonNote={session.windwave.typhoonNote}
              />
            </div>
          )}
        </div>
      </div>
    ) : panel === "eddy" ? (
      <div className="ocean-dashboard__main ocean-dashboard__main--l1 ocean-dashboard__main--l1-eddy-chrome">
        <div className="ocean-dashboard__l1-strip-col">
          {SHOW_HYDRO_UI ? (
            <CollapsedStrip title="水文" accent="sky" onClick={() => goL1("hydro")} />
          ) : (
            <CollapsedStrip
              title="台风查询"
              accent="emerald"
              hint="打开台风查询页"
              onClick={() => navigate(`${modeBase(mode)}/typhoon-kb`)}
            />
          )}
          <CollapsedStrip title="风浪" accent="amber" onClick={() => goL1("windwave")} />
        </div>
      </div>
    ) : null;

  return (
    <div className="ocean-dashboard">
      <div style={{ marginBottom: 10 }}>
        <h1 className="ocean-dashboard__title">{title}</h1>
        <p className="ocean-dashboard__caption">{caption}</p>
      </div>

      <DataSourceBar
        mode={mode}
        offlineNcPath={mode === "offline" ? session.ncPath : null}
        hideOfflineUpload={hideOfflineUploadInBar}
        onOfflineNcUploaded={
          mode === "offline"
            ? (p) => {
                session.setNcPath(p);
                if (SHOW_HYDRO_UI) session.appendHydroBuffer(p);
              }
            : undefined
        }
        uploadBusy={uploadBusy}
        onUploadBusy={setUploadBusy}
        globalErr={globalErr}
        onGlobalErr={setGlobalErr}
        rtStatus={mode === "realtime" && !rtArmed ? "idle" : rt.status}
        rtFingerprint={rt.fingerprint}
        rtError={rt.error}
        rtIntervalSec={rtInterval}
        onRtIntervalSec={setRtInterval}
        rtArmed={rtArmed}
        onRtArmAndRefresh={async () => {
          setRtArmed(true);
          await rt.refresh();
        }}
        rtConnector={rt.connector}
        rtNewFile={rt.newFileDetected}
      />

      <div className="ocean-dashboard__task-strip" aria-label="任务状态与结果导出">
        <div>
          <strong>任务状态</strong>
          <span>
            {session.ncPath
              ? pipelineArmed
                ? "已接入 NetCDF，模块按需自动分析"
                : "已上传 NC，待确认裁剪后运行分析"
              : "等待上传 NetCDF 文件"}
          </span>
        </div>
        <div>
          <strong>分析结果</strong>
          <span>
            风浪 {session.windwave.anomalyLevel ?? "待生成"} · 台风候选 {(session.windwave.typhoonCandidates ?? []).length} · 涡旋{" "}
            {session.eddyChannelMode} · 任务 {session.eddyHistory.length}
          </span>
        </div>
        <div>
          <strong>导出与回溯</strong>
          <button type="button" onClick={exportSessionSummary} disabled={!session.ncPath}>
            导出 JSON 摘要
          </button>
          <button type="button" onClick={() => goL1("eddy")} disabled={session.eddyHistory.length === 0}>
            查看涡旋历史
          </button>
        </div>
      </div>

      <div className={`ocean-dashboard__stage ${panel ? `ocean-dashboard__stage--l1 ocean-dashboard__stage--l1-${panel}` : ""}`}>
        <div className={homeLayerClass}>{homeMain}</div>
        {l1Main}
      </div>
    </div>
  );
}
