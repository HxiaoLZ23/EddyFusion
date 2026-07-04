import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { CurveDataMap } from "./HydroRotatingCurves";

export type OceanMode = "offline" | "realtime";
/** 论文推理固定三通道；7ch 仅仓库实验附录，UI 不提供 */
export type EddyChannelMode = "3ch";

export type EddyJobHistoryRow = {
  id: string;
  at: string;
  ncLabel: string;
  status: "success" | "failed" | "running";
  nFrames?: number;
  message?: string;
  channelMode?: EddyChannelMode;
};

export type EddyDetectionFrameRow = {
  time: string;
  peak_score: number;
  max_conf?: number;
  mean_conf?: number;
  status: "hit" | "miss" | string;
  count?: number;
};

export type HydroHeatmapSnapshot = {
  lons: number[];
  lats: number[];
  values: number[][];
  feature: string;
  kind: string;
  lead: number;
  value_unit?: string;
  vmin?: number;
  vmax?: number;
} | null;

export type HydroSnapshot = {
  curveData: CurveDataMap | null;
  featureNames: string[];
  featureUnits?: Record<string, string>;
  meta: { T_hat?: number; T_need?: number; buffer_sufficient?: boolean } | null;
  heatmap: HydroHeatmapSnapshot;
};

export type TyphoonCandidateRow = {
  event_id?: string;
  id?: string;
  name?: string;
  start_time?: string;
  end_time?: string;
  score?: number;
  dtw_distance?: number;
  wind_level?: string;
  wind_track_mps?: number[];
  wind_track_kt?: number[];
  series_source?: string;
  peak_wind_kt?: number;
  intensity_level?: string;
};

export type TyphoonQueryBox = {
  start_time?: string;
  end_time?: string;
  lon_min?: number;
  lon_max?: number;
  lat_min?: number;
  lat_max?: number;
};

export type WindWaveSeriesPoint = {
  step: number;
  wind_observed: number;
  wind_predicted: number;
  wave_observed: number;
  wave_predicted: number;
};

export type WindwaveSnapshot = {
  reportText: string | null;
  anomalyLevel?: string;
  anomalyIndex?: number | string;
  windWaveSeries?: WindWaveSeriesPoint[];
  typhoonNote?: string | null;
  typhoonCandidates?: TyphoonCandidateRow[];
  typhoonEventsPath?: string | null;
  typhoonQuery?: TyphoonQueryBox | null;
  typhoonRetrieval?: Record<string, unknown> | null;
};

export type WindwaveLlmSnapshot = {
  summaryAnomaly?: string;
  impact?: string;
  historicalAnalogy?: string;
  actions?: string[];
  error?: string | null;
};

export type ModeSlice = {
  ncPath: string | null;
  /** 为 true 时监测总览才自动跑涡旋 MP4 / 风浪报告；勾选裁剪时上传后保持 false 直至确认裁剪 */
  pipelineArmed: boolean;
  eddyChannelMode: EddyChannelMode;
  hydroBufferPaths: string[];
  hydro: HydroSnapshot;
  windwave: WindwaveSnapshot;
  windwaveLlm: WindwaveLlmSnapshot;
  eddyHistory: EddyJobHistoryRow[];
  eddyDetectionFrames: EddyDetectionFrameRow[];
};

function createEmptySlice(): ModeSlice {
  return {
    ncPath: null,
    pipelineArmed: false,
    eddyChannelMode: "3ch",
    hydroBufferPaths: [],
    hydro: { curveData: null, featureNames: [], meta: null, heatmap: null },
    windwave: { reportText: null },
    windwaveLlm: {},
    eddyHistory: [],
    eddyDetectionFrames: [],
  };
}

type PatchFn = (prev: ModeSlice) => ModeSlice;

type OceanSessionContextValue = {
  offline: ModeSlice;
  realtime: ModeSlice;
  patchSlice: (mode: OceanMode, patch: Partial<ModeSlice> | PatchFn) => void;
  resetSlice: (mode: OceanMode) => void;
  /** 记录当前顶栏模式；从离线切到实时时清空实时会话，避免 latest 读到刚上传的 NC 后自动出结果 */
  enterDashboard: (mode: OceanMode) => void;
};

const noop = () => {};

const defaultCtx: OceanSessionContextValue = {
  offline: createEmptySlice(),
  realtime: createEmptySlice(),
  patchSlice: noop,
  resetSlice: noop,
  enterDashboard: noop,
};

const OceanSessionContext = createContext<OceanSessionContextValue>(defaultCtx);

export function OceanSessionProvider({ children }: { children: ReactNode }) {
  const [offline, setOffline] = useState<ModeSlice>(createEmptySlice);
  const [realtime, setRealtime] = useState<ModeSlice>(createEmptySlice);
  const activeDashboardMode = useRef<OceanMode | null>(null);

  const patchSlice = useCallback((mode: OceanMode, patch: Partial<ModeSlice> | PatchFn) => {
    const setter = mode === "offline" ? setOffline : setRealtime;
    setter((prev) => (typeof patch === "function" ? patch(prev) : { ...prev, ...patch }));
  }, []);

  const resetSlice = useCallback((mode: OceanMode) => {
    if (mode === "offline") setOffline(createEmptySlice());
    else setRealtime(createEmptySlice());
  }, []);

  const enterDashboard = useCallback(
    (mode: OceanMode) => {
      const prev = activeDashboardMode.current;
      activeDashboardMode.current = mode;
      if (mode === "realtime" && prev === "offline") {
        setRealtime(createEmptySlice());
      }
    },
    [],
  );

  const value = useMemo(
    () => ({ offline, realtime, patchSlice, resetSlice, enterDashboard }),
    [offline, realtime, patchSlice, resetSlice, enterDashboard],
  );

  return <OceanSessionContext.Provider value={value}>{children}</OceanSessionContext.Provider>;
}

/** 顶栏切换离线/实时时登记模式（触发跨模式清空等） */
export function useDashboardModeLifecycle(mode: OceanMode) {
  const { enterDashboard } = useContext(OceanSessionContext);
  useEffect(() => {
    enterDashboard(mode);
  }, [mode, enterDashboard]);
}

/** 按离线 / 实时隔离的会话数据（NC 路径、水文缓冲、曲线、风浪、涡旋等） */
export function useOceanSession(mode: OceanMode) {
  const { offline, realtime, patchSlice } = useContext(OceanSessionContext);
  const slice = mode === "offline" ? offline : realtime;

  const setNcPath = useCallback(
    (p: string | null) => patchSlice(mode, { ncPath: p }),
    [mode, patchSlice],
  );

  const setPipelineArmed = useCallback(
    (armed: boolean) => patchSlice(mode, { pipelineArmed: armed }),
    [mode, patchSlice],
  );

  const setHydroBufferPaths = useCallback(
    (paths: string[]) => patchSlice(mode, { hydroBufferPaths: paths }),
    [mode, patchSlice],
  );

  const appendHydroBuffer = useCallback(
    (p: string) => {
      patchSlice(mode, (prev) =>
        prev.hydroBufferPaths.includes(p) ? prev : { ...prev, hydroBufferPaths: [...prev.hydroBufferPaths, p] },
      );
    },
    [mode, patchSlice],
  );

  const removeHydroBufferAt = useCallback(
    (index: number) => {
      patchSlice(mode, (prev) => ({
        ...prev,
        hydroBufferPaths: prev.hydroBufferPaths.filter((_, i) => i !== index),
      }));
    },
    [mode, patchSlice],
  );

  const clearHydroBuffer = useCallback(() => {
    patchSlice(mode, { hydroBufferPaths: [] });
  }, [mode, patchSlice]);

  const setHydro = useCallback(
    (v: HydroSnapshot) => patchSlice(mode, { hydro: v }),
    [mode, patchSlice],
  );

  const setWindwave = useCallback(
    (v: WindwaveSnapshot) => patchSlice(mode, { windwave: v }),
    [mode, patchSlice],
  );

  const setWindwaveLlm = useCallback(
    (v: WindwaveLlmSnapshot) => patchSlice(mode, { windwaveLlm: v }),
    [mode, patchSlice],
  );

  const appendEddyJob = useCallback(
    (row: Omit<EddyJobHistoryRow, "id" | "at"> & { id?: string; at?: string }) => {
      const id = row.id ?? `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const at = row.at ?? new Date().toISOString();
      const full: EddyJobHistoryRow = { ...row, id, at };
      patchSlice(mode, (prev) => ({
        ...prev,
        eddyHistory: [full, ...prev.eddyHistory].slice(0, 10),
      }));
    },
    [mode, patchSlice],
  );

  const setEddyDetectionFrames = useCallback(
    (rows: EddyDetectionFrameRow[]) => patchSlice(mode, { eddyDetectionFrames: rows }),
    [mode, patchSlice],
  );

  return {
    ncPath: slice.ncPath,
    pipelineArmed: slice.pipelineArmed,
    setNcPath,
    setPipelineArmed,
    /** @deprecated 使用 ncPath */
    offlineNcPath: slice.ncPath,
    /** @deprecated 使用 setNcPath */
    setOfflineNcPath: setNcPath,
    eddyChannelMode: slice.eddyChannelMode,
    hydroBufferPaths: slice.hydroBufferPaths,
    setHydroBufferPaths,
    appendHydroBuffer,
    removeHydroBufferAt,
    clearHydroBuffer,
    hydro: slice.hydro,
    setHydro,
    windwave: slice.windwave,
    setWindwave,
    windwaveLlm: slice.windwaveLlm,
    setWindwaveLlm,
    eddyHistory: slice.eddyHistory,
    appendEddyJob,
    eddyDetectionFrames: slice.eddyDetectionFrames,
    setEddyDetectionFrames,
  };
}

/** @deprecated 请使用 useOceanSession('offline' | 'realtime') */
export function useOfflineSession() {
  return useOceanSession("offline");
}

export type OfflineSessionValue = ReturnType<typeof useOceanSession>;
