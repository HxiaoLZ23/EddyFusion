import type { EddyDetectionFrameRow, EddyJobHistoryRow } from "./offlineSession";

type Props = {
  rows: EddyJobHistoryRow[];
  detectionFrames: EddyDetectionFrameRow[];
  onClose: () => void;
};

function shortName(path: string): string {
  const p = path.replace(/\\/g, "/");
  const i = p.lastIndexOf("/");
  return i >= 0 ? p.slice(i + 1) : p;
}

/** 规划 §6.2：右侧双列 — 按帧识别 | 生成任务 */
export function EddyHistoryDrawer({ rows, detectionFrames, onClose }: Props) {
  return (
    <aside className="ocean-dashboard__eddy-drawer" aria-label="涡旋生成历史">
      <div className="ocean-dashboard__eddy-drawer-head">
        <strong>涡旋 · 生成历史</strong>
        <button type="button" className="ocean-dashboard__eddy-drawer-close" onClick={onClose}>
          收起
        </button>
      </div>
      <div className="ocean-dashboard__eddy-drawer-grid">
        <section className="ocean-dashboard__eddy-drawer-section">
          <h6 className="ocean-dashboard__eddy-drawer-section-title">识别框 · 按帧（当前视频）</h6>
          <p className="ocean-dashboard__eddy-drawer-caption">
            max_conf 为当帧最高框分；mean_conf 为当帧平均（与视频角标 0.xx 对应）
          </p>
          <div className="ocean-dashboard__eddy-drawer-table-wrap">
            <table className="ocean-dashboard__eddy-drawer-table">
              <thead>
                <tr>
                  <th>time</th>
                  <th>max</th>
                  <th>mean</th>
                  <th>status</th>
                </tr>
              </thead>
              <tbody>
                {detectionFrames.length === 0 ? (
                  <tr>
                    <td colSpan={4} style={{ color: "#64748b", textAlign: "center", padding: "12px 6px" }}>
                      生成带框视频后显示各帧检测摘要
                    </td>
                  </tr>
                ) : (
                  detectionFrames.map((r, i) => (
                    <tr key={`${r.time}-${i}`}>
                      <td className="ocean-dashboard__eddy-drawer-time" title={r.time}>
                        {r.time}
                      </td>
                      <td>
                        {Number.isFinite(r.max_conf ?? r.peak_score)
                          ? Number(r.max_conf ?? r.peak_score).toFixed(3)
                          : "—"}
                      </td>
                      <td>
                        {typeof r.mean_conf === "number" && Number.isFinite(r.mean_conf)
                          ? r.mean_conf.toFixed(3)
                          : "—"}
                      </td>
                      <td>
                        <span
                          className={`ocean-dashboard__eddy-status ocean-dashboard__eddy-status--${r.status === "hit" ? "success" : "failed"}`}
                        >
                          {r.status}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="ocean-dashboard__eddy-drawer-section">
          <h6 className="ocean-dashboard__eddy-drawer-section-title">生成任务（本会话）</h6>
          <p className="ocean-dashboard__eddy-drawer-caption">最近 10 次双路 MP4 请求</p>
          <div className="ocean-dashboard__eddy-drawer-table-wrap">
            <table className="ocean-dashboard__eddy-drawer-table">
              <thead>
                <tr>
                  <th>时间</th>
                  <th>NC</th>
                  <th>状态</th>
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={3} style={{ color: "#64748b", textAlign: "center", padding: "12px 6px" }}>
                      尚无记录
                    </td>
                  </tr>
                ) : (
                  rows.map((r) => (
                    <tr key={r.id}>
                      <td className="ocean-dashboard__eddy-drawer-time">{new Date(r.at).toLocaleString()}</td>
                      <td title={r.ncLabel}>{shortName(r.ncLabel)}</td>
                      <td>
                        <span className={`ocean-dashboard__eddy-status ocean-dashboard__eddy-status--${r.status}`}>
                          {r.status}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </aside>
  );
}
