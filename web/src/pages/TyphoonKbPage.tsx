import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  fetchTyphoonDemoCases,
  fetchTyphoonKbDefaults,
  fetchTyphoonKbEvents,
  fetchTyphoonKbStatus,
  friendlyTyphoonLevel,
  postTyphoonKbQuery,
  type TyphoonCandidate,
  type TyphoonEventRow,
  type TyphoonKbDefaults,
  type TyphoonKbStatus,
} from "../adapters/typhoonKbAdapter";
import type { OceanMode } from "../dashboard/offlineSession";
import "./typhoon-kb-page.css";

type TabId = "query" | "browse" | "cases";

type Props = {
  mode: OceanMode;
};

type LocationState = {
  prefill?: Partial<TyphoonKbDefaults>;
};

function modeHome(mode: OceanMode): string {
  return mode === "offline" ? "/offline" : "/realtime";
}

export function TyphoonKbPage({ mode }: Props) {
  const location = useLocation();
  const prefill = (location.state as LocationState | null)?.prefill;

  const [tab, setTab] = useState<TabId>("query");
  const [status, setStatus] = useState<TyphoonKbStatus | null>(null);
  const [defaults, setDefaults] = useState<TyphoonKbDefaults | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  const [startTime, setStartTime] = useState("");
  const [endTime, setEndTime] = useState("");
  const [lonMin, setLonMin] = useState(0);
  const [lonMax, setLonMax] = useState(0);
  const [latMin, setLatMin] = useState(0);
  const [latMax, setLatMax] = useState(0);
  const [topK, setTopK] = useState(10);
  const [eventsPath, setEventsPath] = useState("");

  const [queryBusy, setQueryBusy] = useState(false);
  const [queryErr, setQueryErr] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<TyphoonCandidate[]>([]);

  const [browseKw, setBrowseKw] = useState("");
  const [browseLevel, setBrowseLevel] = useState<string[]>([]);
  const [browseSeason, setBrowseSeason] = useState<string[]>([]);
  const [browsePage, setBrowsePage] = useState(1);
  const [browseItems, setBrowseItems] = useState<TyphoonEventRow[]>([]);
  const [browseTotal, setBrowseTotal] = useState(0);
  const [browseMaxPage, setBrowseMaxPage] = useState(1);
  const [browseFacets, setBrowseFacets] = useState<{ levels: string[]; seasons: string[] }>({
    levels: [],
    seasons: [],
  });
  const [browseBusy, setBrowseBusy] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<TyphoonEventRow | null>(null);

  const [demoCases, setDemoCases] = useState<unknown[]>([]);

  useEffect(() => {
    let cancelled = false;
    setLoadErr(null);
    Promise.all([fetchTyphoonKbStatus(), fetchTyphoonKbDefaults()])
      .then(([st, def]) => {
        if (cancelled) return;
        setStatus(st);
        setDefaults(def);
        const p = prefill ?? {};
        setStartTime(String(p.start_time ?? def.start_time));
        setEndTime(String(p.end_time ?? def.end_time));
        setLonMin(Number(p.lon_min ?? def.lon_min));
        setLonMax(Number(p.lon_max ?? def.lon_max));
        setLatMin(Number(p.lat_min ?? def.lat_min));
        setLatMax(Number(p.lat_max ?? def.lat_max));
        setTopK(Number(p.top_k ?? def.top_k));
        setEventsPath(String(p.events_json_path ?? def.events_json_path));
      })
      .catch((e) => {
        if (!cancelled) setLoadErr(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const runQuery = useCallback(async () => {
    setQueryBusy(true);
    setQueryErr(null);
    setCandidates([]);
    try {
      const out = await postTyphoonKbQuery({
        start_time: startTime,
        end_time: endTime,
        lon_min: lonMin,
        lon_max: lonMax,
        lat_min: latMin,
        lat_max: latMax,
        top_k: topK,
        events_json_path: eventsPath || undefined,
      });
      setCandidates(out.candidates);
    } catch (e) {
      setQueryErr(e instanceof Error ? e.message : String(e));
    } finally {
      setQueryBusy(false);
    }
  }, [startTime, endTime, lonMin, lonMax, latMin, latMax, topK, eventsPath]);

  const loadBrowse = useCallback(async () => {
    setBrowseBusy(true);
    try {
      const out = await fetchTyphoonKbEvents({
        page: browsePage,
        page_size: 20,
        keyword: browseKw,
        level: browseLevel.join(","),
        season: browseSeason.join(","),
        events_json_path: eventsPath || undefined,
      });
      setBrowseItems(out.items);
      setBrowseTotal(out.total);
      setBrowseMaxPage(out.max_page);
      setBrowseFacets(out.facets);
      setSelectedEvent(out.items[0] ?? null);
    } catch (e) {
      setLoadErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBrowseBusy(false);
    }
  }, [browsePage, browseKw, browseLevel, browseSeason, eventsPath]);

  useEffect(() => {
    if (tab !== "browse" || !status?.ready) return;
    void loadBrowse();
  }, [tab, status?.ready, loadBrowse]);

  useEffect(() => {
    if (tab !== "cases" || !defaults) return;
    let cancelled = false;
    void fetchTyphoonDemoCases(defaults.demo_cases_path)
      .then((r) => {
        if (!cancelled) setDemoCases(r.cases);
      })
      .catch(() => {
        if (!cancelled) setDemoCases([]);
      });
    return () => {
      cancelled = true;
    };
  }, [tab, defaults]);

  const statusLine = useMemo(() => {
    if (!status) return "—";
    return status.ready
      ? `就绪 · ${status.events_count} 条 · ${status.source ?? "本地索引"}`
      : "索引缺失（API 启动时会尝试 seed 演示库）";
  }, [status]);

  return (
    <div className="typhoon-kb-page">
      <header className="typhoon-kb-page__header">
        <div>
          <Link className="typhoon-kb-page__back" to={modeHome(mode)}>
            ← 返回{mode === "offline" ? "离线" : "实时"}同屏
          </Link>
          <h1 className="typhoon-kb-page__title">台风查询</h1>
          <p className="typhoon-kb-page__caption">
            时空检索历史台风事件（IBTrACS 构建或演示索引）；与风浪异常联动口径一致，候选经 bbox 重叠与 DTW 重排。
          </p>
        </div>
        <div className="typhoon-kb-page__status-pill">{statusLine}</div>
      </header>

      {loadErr && <p className="typhoon-kb-page__err">{loadErr}</p>}

      <nav className="typhoon-kb-page__tabs" role="tablist">
        {(
          [
            ["query", "快速检索"],
            ["browse", "历史浏览"],
            ["cases", "联动案例"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            className={`typhoon-kb-page__tab ${tab === id ? "typhoon-kb-page__tab--active" : ""}`}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      {tab === "query" && (
        <section className="typhoon-kb-page__section">
          <div className="typhoon-kb-page__form-grid">
            <label>
              开始时间
              <input value={startTime} onChange={(e) => setStartTime(e.target.value)} />
            </label>
            <label>
              结束时间
              <input value={endTime} onChange={(e) => setEndTime(e.target.value)} />
            </label>
            <label>
              lon_min
              <input type="number" step="0.1" value={lonMin} onChange={(e) => setLonMin(Number(e.target.value))} />
            </label>
            <label>
              lon_max
              <input type="number" step="0.1" value={lonMax} onChange={(e) => setLonMax(Number(e.target.value))} />
            </label>
            <label>
              lat_min
              <input type="number" step="0.1" value={latMin} onChange={(e) => setLatMin(Number(e.target.value))} />
            </label>
            <label>
              lat_max
              <input type="number" step="0.1" value={latMax} onChange={(e) => setLatMax(Number(e.target.value))} />
            </label>
            <label>
              Top-K
              <input
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </label>
            <label className="typhoon-kb-page__span2">
              事件索引路径
              <input value={eventsPath} onChange={(e) => setEventsPath(e.target.value)} />
            </label>
          </div>
          <button type="button" className="typhoon-kb-page__primary" disabled={queryBusy} onClick={() => void runQuery()}>
            {queryBusy ? "检索中…" : "查询台风候选"}
          </button>
          {queryErr && <p className="typhoon-kb-page__err">{queryErr}</p>}
          {candidates.length > 0 && (
            <div className="typhoon-kb-page__table-wrap">
              <p className="typhoon-kb-page__result-count">命中 {candidates.length} 条</p>
              <table className="typhoon-kb-page__table">
                <thead>
                  <tr>
                    <th>事件</th>
                    <th>名称</th>
                    <th>起止</th>
                    <th>强度</th>
                    <th>时窗重叠(h)</th>
                    <th>区域重叠</th>
                    <th>分数</th>
                  </tr>
                </thead>
                <tbody>
                  {candidates.map((c, i) => (
                    <tr key={`${c.event_id ?? i}`}>
                      <td>{c.event_id ?? "—"}</td>
                      <td>{c.name ?? "—"}</td>
                      <td className="typhoon-kb-page__time">
                        {c.start_time && c.end_time ? `${c.start_time} ~ ${c.end_time}` : "—"}
                      </td>
                      <td>{friendlyTyphoonLevel(c.intensity_level)}</td>
                      <td>{c.time_overlap_hours ?? "—"}</td>
                      <td>{c.bbox_overlap_ratio ?? "—"}</td>
                      <td>{c.score != null ? Number(c.score).toFixed(3) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {!queryBusy && !queryErr && candidates.length === 0 && (
            <p className="typhoon-kb-page__muted">填写时空范围后点击「查询台风候选」。</p>
          )}
        </section>
      )}

      {tab === "browse" && (
        <section className="typhoon-kb-page__section">
          <div className="typhoon-kb-page__browse-filters">
            <input
              placeholder="关键词（事件 ID / 名称）"
              value={browseKw}
              onChange={(e) => {
                setBrowseKw(e.target.value);
                setBrowsePage(1);
              }}
            />
            <select
              multiple
              value={browseLevel}
              onChange={(e) => {
                setBrowseLevel(Array.from(e.target.selectedOptions, (o) => o.value));
                setBrowsePage(1);
              }}
              size={3}
            >
              {browseFacets.levels.map((lv) => (
                <option key={lv} value={lv}>
                  {friendlyTyphoonLevel(lv)} ({lv})
                </option>
              ))}
            </select>
            <select
              multiple
              value={browseSeason}
              onChange={(e) => {
                setBrowseSeason(Array.from(e.target.selectedOptions, (o) => o.value));
                setBrowsePage(1);
              }}
              size={3}
            >
              {browseFacets.seasons.map((sy) => (
                <option key={sy} value={sy}>
                  {sy}
                </option>
              ))}
            </select>
            <button type="button" className="typhoon-kb-page__secondary" disabled={browseBusy} onClick={() => void loadBrowse()}>
              刷新列表
            </button>
          </div>
          <p className="typhoon-kb-page__muted">
            共 {browseTotal} 条 · 第 {browsePage}/{browseMaxPage} 页
          </p>
          {browseBusy ? (
            <p className="typhoon-kb-page__muted">加载中…</p>
          ) : (
            <>
              <div className="typhoon-kb-page__table-wrap">
                <table className="typhoon-kb-page__table">
                  <thead>
                    <tr>
                      <th>事件 ID</th>
                      <th>名称</th>
                      <th>年份</th>
                      <th>强度</th>
                      <th>起止</th>
                      <th>峰值风速(kt)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {browseItems.map((e) => (
                      <tr
                        key={String(e.event_id)}
                        className={selectedEvent?.event_id === e.event_id ? "typhoon-kb-page__row--sel" : ""}
                        onClick={() => setSelectedEvent(e)}
                      >
                        <td>{e.event_id}</td>
                        <td>{e.name ?? "—"}</td>
                        <td>{e.season ?? "—"}</td>
                        <td>{friendlyTyphoonLevel(e.intensity_level)}</td>
                        <td className="typhoon-kb-page__time">
                          {e.start_time && e.end_time ? `${e.start_time} ~ ${e.end_time}` : "—"}
                        </td>
                        <td>{e.peak_wind_kt ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="typhoon-kb-page__pager">
                <button
                  type="button"
                  disabled={browsePage <= 1}
                  onClick={() => setBrowsePage((p) => Math.max(1, p - 1))}
                >
                  上一页
                </button>
                <button
                  type="button"
                  disabled={browsePage >= browseMaxPage}
                  onClick={() => setBrowsePage((p) => p + 1)}
                >
                  下一页
                </button>
              </div>
              {selectedEvent && (
                <aside className="typhoon-kb-page__detail">
                  <h3>事件详情 · {selectedEvent.event_id}</h3>
                  <ul>
                    <li>名称：{selectedEvent.name ?? "—"}</li>
                    <li>年份：{selectedEvent.season ?? "—"}</li>
                    <li>强度：{friendlyTyphoonLevel(selectedEvent.intensity_level)}</li>
                    <li>
                      时间：{selectedEvent.start_time} ~ {selectedEvent.end_time}
                    </li>
                    <li>
                      包围盒：lon[{selectedEvent.lon_min}, {selectedEvent.lon_max}] lat[{selectedEvent.lat_min},{" "}
                      {selectedEvent.lat_max}]
                    </li>
                    <li>
                      中心：({selectedEvent.center_lon}, {selectedEvent.center_lat})
                    </li>
                    <li>轨迹点数：{selectedEvent.n_points ?? "—"}</li>
                    <li>峰值风速(kt)：{selectedEvent.peak_wind_kt ?? "—"}</li>
                  </ul>
                </aside>
              )}
            </>
          )}
        </section>
      )}

      {tab === "cases" && (
        <section className="typhoon-kb-page__section">
          {demoCases.length === 0 ? (
            <p className="typhoon-kb-page__muted">
              未找到 demo_cases.json，可运行 <code>scripts/demo_typhoon_kb_cases.py</code>。
            </p>
          ) : (
            <ul className="typhoon-kb-page__case-list">
              {demoCases.map((raw, i) => {
                const c = raw as {
                  case_id?: string;
                  query?: Record<string, unknown>;
                  results?: TyphoonCandidate[];
                };
                return (
                  <li key={c.case_id ?? i} className="typhoon-kb-page__case">
                    <h3>{c.case_id ?? `案例 ${i + 1}`}</h3>
                    {c.query && (
                      <p className="typhoon-kb-page__muted">
                        {String(c.query.start_time)} ~ {String(c.query.end_time)} · lon[{String(c.query.lon_min)},
                        {String(c.query.lon_max)}] lat[{String(c.query.lat_min)},{String(c.query.lat_max)}]
                      </p>
                    )}
                    {(c.results?.length ?? 0) > 0 ? (
                      <ul>
                        {c.results!.slice(0, 5).map((r, j) => (
                          <li key={j}>
                            {r.event_id} {r.name ? `(${r.name})` : ""} · {r.summary ?? r.score}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="typhoon-kb-page__muted">无命中事件</p>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      )}
    </div>
  );
}
