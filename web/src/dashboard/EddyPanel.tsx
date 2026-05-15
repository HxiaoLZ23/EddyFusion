import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { eddyPreviewUrl, postEddyDualMp4 } from "../adapters/eddyDualMp4Adapter";

type Props = {
  /** 仓库相对路径或白名单绝对路径 */
  ncPath: string;
  /** 离线：上传成功后自动请求双路 MP4 */
  autoGenerate?: boolean;
};

/** 左栏：双路 MP4（底图 / 带框）+ 时间戳条；仅主路驱动同步，避免双路 timeupdate 互抢导致控件闪烁。 */
export function EddyPanel({ ncPath, autoGenerate = false }: Props) {
  const baseRef = useRef<HTMLVideoElement | null>(null);
  const annRef = useRef<HTMLVideoElement | null>(null);
  const syncingRef = useRef(false);
  const lastBarIdx = useRef(-1);

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [baseFile, setBaseFile] = useState<string | null>(null);
  const [annFile, setAnnFile] = useState<string | null>(null);
  const [fps, setFps] = useState(1);
  const [timeLabels, setTimeLabels] = useState<string[]>([]);
  const [truncated, setTruncated] = useState(false);
  const [nFrames, setNFrames] = useState(0);
  const [barText, setBarText] = useState("请先选择 NC（上传或实时 latest）");

  const baseSrc = useMemo(() => (baseFile ? eddyPreviewUrl(baseFile) : ""), [baseFile]);
  const annSrc = useMemo(() => (annFile ? eddyPreviewUrl(annFile) : ""), [annFile]);

  const frameIndex = useCallback(
    (t: number) => {
      const n = timeLabels.length;
      if (n <= 0) return 0;
      const fi = Math.floor(Math.max(0, t) * fps + 1e-6);
      return Math.min(n - 1, Math.max(0, fi));
    },
    [fps, timeLabels.length],
  );

  const tickBar = useCallback(
    (t: number) => {
      const idx = frameIndex(t);
      if (idx === lastBarIdx.current) return;
      lastBarIdx.current = idx;
      setBarText(timeLabels[idx] ?? `帧 ${idx}`);
    },
    [frameIndex, timeLabels],
  );

  /** 仅主路（上）为 leader：校正下路时间并更新时间条，不在下路挂 timeupdate，避免与主路循环 seek。 */
  const onBaseTimeUpdate = useCallback(() => {
    const b = baseRef.current;
    const a = annRef.current;
    if (!b || !a || syncingRef.current) return;
    if (Math.abs(a.currentTime - b.currentTime) > 0.06) {
      syncingRef.current = true;
      a.currentTime = b.currentTime;
      queueMicrotask(() => {
        syncingRef.current = false;
      });
    }
    tickBar(b.currentTime);
  }, [tickBar]);

  const onBaseSeeked = useCallback(() => {
    const b = baseRef.current;
    const a = annRef.current;
    if (!b || !a || syncingRef.current) return;
    syncingRef.current = true;
    a.currentTime = b.currentTime;
    queueMicrotask(() => {
      syncingRef.current = false;
    });
    tickBar(b.currentTime);
  }, [tickBar]);

  const playPair = useCallback(() => {
    const b = baseRef.current;
    const a = annRef.current;
    if (!b || !a) return;
    syncingRef.current = true;
    a.currentTime = b.currentTime;
    queueMicrotask(() => {
      syncingRef.current = false;
    });
    void b.play().catch(() => {});
    void a.play().catch(() => {});
  }, []);

  const autoplayKeyRef = useRef<string>("");

  const onBuild = useCallback(async () => {
    if (!ncPath.trim()) {
      setErr("未选择 NC");
      return;
    }
    setErr(null);
    setBusy(true);
    setBaseFile(null);
    setAnnFile(null);
    setTimeLabels([]);
    setNFrames(0);
    lastBarIdx.current = -1;
    setBarText("编码中…");
    try {
      const out = await postEddyDualMp4({
        nc_path: ncPath,
        fps: 1,
        max_frames: 120,
      });
      setBaseFile(out.preview_base);
      setAnnFile(out.preview_annotated);
      setFps(Number(out.fps) || 1);
      setTimeLabels(out.time_labels ?? []);
      setTruncated(Boolean(out.truncated));
      setNFrames(Number(out.n_frames) || 0);
      setBarText(out.time_labels?.[0] ?? "就绪");
      lastBarIdx.current = 0;
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setBarText("生成失败");
    } finally {
      setBusy(false);
    }
  }, [ncPath, autoGenerate]);

  useEffect(() => {
    setBaseFile(null);
    setAnnFile(null);
    setTimeLabels([]);
    setTruncated(false);
    setNFrames(0);
    setErr(null);
    lastBarIdx.current = -1;
    if (!ncPath.trim()) {
      setBarText("请先选择 NC（上传或实时 latest）");
    } else if (autoGenerate) {
      setBarText("将自动生成双路视频…");
    } else {
      setBarText("可点击「生成双路视频」");
    }
  }, [ncPath, autoGenerate]);

  useEffect(() => {
    if (!autoGenerate || !ncPath.trim()) return;
    void onBuild();
  }, [autoGenerate, ncPath, onBuild]);

  /** 两路 src 就绪后自动播放（均已 muted）；同一对 URL 只触发一次，避免 StrictMode 双调用反复 play。 */
  useEffect(() => {
    if (!baseSrc || !annSrc) {
      autoplayKeyRef.current = "";
      return;
    }
    const key = `${baseSrc}|${annSrc}`;
    if (autoplayKeyRef.current === key) return;
    const t = window.setTimeout(() => {
      autoplayKeyRef.current = key;
      playPair();
    }, 200);
    return () => window.clearTimeout(t);
  }, [baseSrc, annSrc, playPair]);

  return (
    <div className="ocean-dashboard__panel" style={{ height: "100%" }}>
      <h3 className="ocean-dashboard__panel-head">涡旋（双路视频）</h3>
      <div className="ocean-dashboard__eddy-toolbar">
        {!autoGenerate && (
          <button type="button" className="ocean-dashboard__eddy-btn" disabled={busy || !ncPath} onClick={() => void onBuild()}>
            {busy ? "生成中…" : "生成双路视频"}
          </button>
        )}
        {autoGenerate && busy && <span className="ocean-dashboard__eddy-hint">自动生成中（长序列会抽样以加速）…</span>}
        {truncated && nFrames > 0 && (
          <span className="ocean-dashboard__eddy-hint" title="为缩短推理与编码时间，对时间维做了截断或均匀抽样">
            预览共 {nFrames} 帧（长序列已抽样 / 截断）
          </span>
        )}
        {err && <span className="ocean-dashboard__eddy-err">{err}</span>}
      </div>
      <div className="ocean-dashboard__timestamp-bar ocean-dashboard__timestamp-bar--eddy">{barText}</div>
      <div className="ocean-dashboard__eddy-videos">
        <div className="ocean-dashboard__video-slot ocean-dashboard__video-slot--live ocean-dashboard__eddy-slot ocean-dashboard__eddy-slot--top">
          <div className="ocean-dashboard__video-slot-label">上 · 底图 / 流场（无检测框）· 播放控制</div>
          {baseSrc ? (
            <video
              ref={baseRef}
              className="ocean-dashboard__eddy-video"
              src={baseSrc}
              controls
              muted
              playsInline
              preload="auto"
              onTimeUpdate={onBaseTimeUpdate}
              onSeeked={onBaseSeeked}
            />
          ) : (
            <div className="ocean-dashboard__video-slot-placeholder">{busy ? "编码中…" : "生成后显示"}</div>
          )}
        </div>
        <div className="ocean-dashboard__video-slot ocean-dashboard__video-slot--live ocean-dashboard__eddy-slot ocean-dashboard__eddy-slot--bottom">
          <div className="ocean-dashboard__video-slot-label">下 · 同帧检测框 / 掩码（与上同步，无独立控件）</div>
          {annSrc ? (
            <video
              ref={annRef}
              className="ocean-dashboard__eddy-video ocean-dashboard__eddy-video--synced"
              src={annSrc}
              controls={false}
              muted
              playsInline
              preload="auto"
            />
          ) : (
            <div className="ocean-dashboard__video-slot-placeholder">{busy ? "编码中…" : "生成后显示"}</div>
          )}
        </div>
      </div>
      <p style={{ fontSize: 11, color: "#94a3b8", margin: "8px 0 0", flexShrink: 0 }}>
        需 NC 含 time 维且 T≥2。请用<strong>上</strong>方播放器拖动进度；下路无控件以免双路控件争抢。服务端对长序列会提高时间步长并默认最多约 32 帧 YOLO 推理，可用环境变量 EDDY_DUAL_MAX_INFER_FRAMES 调整。
      </p>
    </div>
  );
}
