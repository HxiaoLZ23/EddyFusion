import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  fetchTyphoonKbStatus,
  friendlyTyphoonLevel,
  type TyphoonKbStatus,
} from "../adapters/typhoonKbAdapter";
import { type OceanMode, useOceanSession, type TyphoonCandidateRow } from "./offlineSession";
import { parseTyphoonDtwMeta, TyphoonDtwMetaBar, windTrackValues, WindTrackSparkline } from "./typhoonDtwUi";

type Props = {
  mode: OceanMode;
};

function candidateId(c: TyphoonCandidateRow): string {
  return String(c.event_id ?? c.id ?? c.name ?? "—");
}

/** 一级同屏：展示风浪联动命中的台风摘要 + 跳转台风查询页 */
export function TyphoonKbPanel({ mode }: Props) {
  const { windwave } = useOceanSession(mode);
  const [status, setStatus] = useState<TyphoonKbStatus | null>(null);
  const [statusErr, setStatusErr] = useState<string | null>(null);

  const candidates = windwave.typhoonCandidates ?? [];
  const dtwMeta = parseTyphoonDtwMeta(windwave.typhoonRetrieval);
  const kbPath = mode === "offline" ? "/offline/typhoon-kb" : "/realtime/typhoon-kb";
  const prefill = windwave.typhoonQuery
    ? {
        ...windwave.typhoonQuery,
        top_k: 10,
        events_json_path: windwave.typhoonEventsPath ?? undefined,
      }
    : undefined;

  useEffect(() => {
    let cancelled = false;
    void fetchTyphoonKbStatus()
      .then((s) => {
        if (!cancelled) {
          setStatus(s);
          setStatusErr(null);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setStatus(null);
          setStatusErr(e instanceof Error ? e.message : String(e));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="ocean-dashboard__panel ocean-dashboard__panel--typhoon">
      <div className="ocean-dashboard__typhoon-head">
        <h3 className="ocean-dashboard__panel-head ocean-dashboard__panel-head--inline">台风查询</h3>
        <Link className="ocean-dashboard__typhoon-go" to={kbPath} state={prefill ? { prefill } : undefined}>
          台风查询 →
        </Link>
      </div>

      <div className="ocean-dashboard__typhoon-status-row">
        {statusErr ? (
          <span className="ocean-dashboard__typhoon-status ocean-dashboard__typhoon-status--err">索引状态未知</span>
        ) : status ? (
          <span
            className={`ocean-dashboard__typhoon-status ${
              status.ready ? "ocean-dashboard__typhoon-status--ok" : "ocean-dashboard__typhoon-status--warn"
            }`}
          >
            {status.ready
              ? `索引就绪 · ${status.events_count} 条事件${status.source ? ` · ${status.source}` : ""}`
              : "索引未构建（演示库将在 API 启动时自动 seed）"}
          </span>
        ) : (
          <span className="ocean-dashboard__typhoon-status">正在检查知识库…</span>
        )}
      </div>

      <div className="ocean-dashboard__panel-body ocean-dashboard__panel-body--typhoon">
        {candidates.length > 0 ? (
          <>
            <TyphoonDtwMetaBar meta={dtwMeta} />
            <p className="ocean-dashboard__typhoon-lead">
              与当前风浪/异常窗口联动的历史台风（Top {Math.min(candidates.length, 5)}）：
            </p>
            <ul className="ocean-dashboard__typhoon-list">
              {candidates.slice(0, 5).map((c, i) => {
                const track = windTrackValues(c);
                return (
                  <li key={`${candidateId(c)}-${i}`} className="ocean-dashboard__typhoon-item">
                    <span className="ocean-dashboard__typhoon-item-id">{candidateId(c)}</span>
                    {c.name ? <span className="ocean-dashboard__typhoon-item-name">{c.name}</span> : null}
                    <span className="ocean-dashboard__typhoon-item-meta">
                      {c.start_time && c.end_time ? `${c.start_time} ~ ${c.end_time}` : "—"}
                      {c.score != null ? ` · 分 ${Number(c.score).toFixed(2)}` : ""}
                      {c.dtw_distance != null ? ` · DTW ${Number(c.dtw_distance).toFixed(3)}` : ""}
                      {track.length > 0 ? ` · 风轨迹 ${track.length} 点` : " · 峰值降级"}
                    </span>
                    {track.length > 1 ? (
                      <span className="ocean-dashboard__typhoon-item-track">
                        <WindTrackSparkline values={track} />
                      </span>
                    ) : null}
                    {(c.wind_level || (c as { intensity_level?: string }).intensity_level) && (
                      <span className="ocean-dashboard__typhoon-item-level">
                        {friendlyTyphoonLevel(
                          String(c.wind_level ?? (c as { intensity_level?: string }).intensity_level),
                        )}
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>
            {windwave.typhoonNote && <p className="ocean-dashboard__typhoon-note">{windwave.typhoonNote}</p>}
          </>
        ) : (
          <p className="ocean-dashboard__typhoon-empty">
            {windwave.reportText
              ? "当前风浪报告未命中可对齐的历史台风（多为时间窗与海区 bbox 无交集）。可在「台风查询」中自定义时空范围检索。"
              : "上传含 u10/v10 或波高的 NC 并生成风浪报告后，此处将显示联动的相似台风事件。"}
          </p>
        )}
        <p className="ocean-dashboard__typhoon-hint">
          完整库：<code>scripts/run_typhoon_kb.ps1</code>；演示索引：<code>seed_typhoon_kb_demo.py</code>
        </p>
      </div>
    </div>
  );
}
