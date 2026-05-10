"""规则模板预警报告。"""

from __future__ import annotations


def _safe(v: object, default: str = "-") -> str:
    if v is None:
        return default
    text = str(v).strip()
    return text if text else default


def render_report(
    *,
    detect_output: dict,
    title: str = "风浪异常预警报告",
) -> str:
    """
    将 run_detect 输出渲染为结构化文本报告。
    """
    anomaly = detect_output.get("anomaly_result", {}) if isinstance(detect_output, dict) else {}
    link = detect_output.get("typhoon_link", {}) if isinstance(detect_output, dict) else {}
    query = link.get("query", {}) if isinstance(link, dict) else {}
    candidates = link.get("candidates", []) if isinstance(link, dict) else []

    lines: list[str] = []
    lines.append(f"【{title}】")
    lines.append("")
    lines.append("一、异常判定")
    lines.append(f"- 异常等级: {_safe(anomaly.get('anomaly_level'))}")
    lines.append(f"- 异常指数: {_safe(anomaly.get('anomaly_index'))}")
    lines.append(f"- 风速残差: {_safe(anomaly.get('wind_residual'))} (z={_safe(anomaly.get('wind_z'))})")
    lines.append(f"- 波高残差: {_safe(anomaly.get('wave_residual'))} (z={_safe(anomaly.get('wave_z'))})")
    lines.append(f"- 判定规则: {_safe(anomaly.get('threshold_rule'), default='3sigma')}")
    if anomaly.get("assessment_note"):
        lines.append(f"- 备注: {_safe(anomaly.get('assessment_note'))}")
    lines.append("")
    lines.append("二、异常时空窗口")
    lines.append(
        f"- 时间窗: {_safe(query.get('start_time'))} ~ {_safe(query.get('end_time'))}"
    )
    lines.append(
        "- 区域: "
        f"lon[{_safe(query.get('lon_min'))}, {_safe(query.get('lon_max'))}], "
        f"lat[{_safe(query.get('lat_min'))}, {_safe(query.get('lat_max'))}]"
    )
    lines.append("")
    lines.append("三、历史台风相似事件 Top-K")
    if not candidates:
        lines.append("- 未检索到候选台风事件。")
    else:
        for idx, item in enumerate(candidates[:5], start=1):
            eid = _safe(item.get("event_id") or item.get("id") or item.get("name"))
            st = _safe(item.get("start_time"))
            et = _safe(item.get("end_time"))
            sim = _safe(item.get("score"))
            dtw = _safe(item.get("dtw_distance"))
            lines.append(
                f"- Top{idx}: {eid} | 时间 {st}~{et} | 时空匹配分 {sim} | DTW距离 {dtw}"
            )
    lines.append("")
    lines.append("四、建议")
    level = str(anomaly.get("anomaly_level", "")).lower()
    if level == "high":
        lines.append("- 建议升级为重点关注事件，启动高频更新与人工复核。")
    elif level == "medium":
        lines.append("- 建议持续跟踪并结合未来 6-12 小时实测数据复核。")
    else:
        lines.append("- 当前为低等级异常，建议常规监测。")
    lines.append("- 建议结合业务阈值与命题方评测口径进行最终判定。")

    return "\n".join(lines)
