from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import streamlit as st

from src.anomaly.eddy_typhoon_bridge import typhoon_query_bbox_from_configs
from src.anomaly.typhoon_kb import QueryBox, query_typhoon_events
from src.utils.config import load_yaml, resolve_path


def _safe_float(v: object, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _read_json(path: str | Path) -> dict[str, Any] | list[Any] | None:
    p = resolve_path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _friendly_level(level: str) -> str:
    mapping = {
        "typhoon": "台风",
        "tropical_storm": "热带风暴",
        "tropical_depression": "热带低压",
        "unknown": "未知",
    }
    return mapping.get(str(level), str(level))


def _default_query_params() -> dict[str, Any]:
    demo_cfg = load_yaml("app/config/demo.yaml")
    ty_cfg = demo_cfg.get("typhoon_link", {}) if isinstance(demo_cfg, dict) else {}

    end_dt = datetime.now()
    win_h = int(_safe_float(ty_cfg.get("default_window_hours"), 240))
    start_dt = end_dt - timedelta(hours=max(1, win_h))
    lon_min, lon_max, lat_min, lat_max = typhoon_query_bbox_from_configs()
    return {
        "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "top_k": int(_safe_float(ty_cfg.get("default_top_k"), 5)),
        "events_json_path": str(ty_cfg.get("events_json_path", "data/processed/anomaly/typhoon_kb/events.json")),
        "demo_cases_path": str(ty_cfg.get("demo_cases_path", "data/processed/anomaly/typhoon_kb/demo_cases.json")),
    }


def _parse_time(raw: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {raw}")


def _render_kb_status(events_path: str) -> None:
    st.subheader("知识库状态")
    idx_path = "data/processed/anomaly/typhoon_index.json"
    idx = _read_json(idx_path)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("events.json", "存在" if resolve_path(events_path).is_file() else "缺失")
    with c2:
        if isinstance(idx, dict):
            st.metric("事件数", str(idx.get("events_count", "unknown")))
        else:
            st.metric("事件数", "unknown")

    if isinstance(idx, dict):
        with st.expander("索引摘要", expanded=False):
            st.json(idx)
    else:
        st.info("未读取到 typhoon_index.json，可先运行 scripts/run_typhoon_kb.*")


def _format_rows_for_user(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "事件ID": r.get("event_id", "-"),
                "名称": r.get("name", ""),
                "起止时间": f"{r.get('start_time','')} ~ {r.get('end_time','')}",
                "强度": _friendly_level(str(r.get("intensity_level", ""))),
                "峰值风速(kt)": r.get("peak_wind_kt", ""),
                "时窗重叠(h)": r.get("time_overlap_hours", ""),
                "区域重叠": r.get("bbox_overlap_ratio", ""),
                "相关分数": r.get("score", ""),
                "事件摘要": r.get("summary", ""),
            }
        )
    return out


def _render_query_panel(defaults: dict[str, Any]) -> None:
    st.subheader("快速检索")
    st.caption("输入时间窗和海域范围，快速返回相关台风事件。")
    col1, col2 = st.columns(2)
    with col1:
        start_time = st.text_input("开始时间", value=defaults["start_time"], key="kb_start_time")
        end_time = st.text_input("结束时间", value=defaults["end_time"], key="kb_end_time")
        top_k = st.slider("候选数量 Top-K", min_value=1, max_value=50, value=int(defaults["top_k"]), key="kb_top_k")
    with col2:
        lon_min = st.number_input("lon_min", value=float(defaults["lon_min"]), step=0.1, key="kb_lon_min")
        lon_max = st.number_input("lon_max", value=float(defaults["lon_max"]), step=0.1, key="kb_lon_max")
        lat_min = st.number_input("lat_min", value=float(defaults["lat_min"]), step=0.1, key="kb_lat_min")
        lat_max = st.number_input("lat_max", value=float(defaults["lat_max"]), step=0.1, key="kb_lat_max")

    events_json_path = st.text_input("事件索引路径", value=defaults["events_json_path"], key="kb_events_json_path")
    run = st.button("查询台风候选事件", type="primary", key="kb_query_btn")
    auto_run = bool(st.session_state.pop("kb_query_autorun", False))
    if not run and not auto_run:
        st.caption("点击“查询台风候选事件”执行检索。")
        return

    try:
        query = QueryBox(
            start_time=_parse_time(start_time),
            end_time=_parse_time(end_time),
            lon_min=float(lon_min),
            lon_max=float(lon_max),
            lat_min=float(lat_min),
            lat_max=float(lat_max),
        )
        rows = query_typhoon_events(events_json_path=events_json_path, query=query, top_k=int(top_k))
    except Exception as e:
        st.error(f"查询失败：{e}")
        return

    if not rows:
        st.warning("未检索到候选事件。")
        return

    st.success(f"检索到 {len(rows)} 个候选事件")
    table = _format_rows_for_user(rows)
    st.dataframe(table, use_container_width=True, hide_index=True)
    with st.expander("查询结果 JSON", expanded=False):
        st.json(rows)


def _render_event_browser(defaults: dict[str, Any]) -> None:
    st.subheader("历史事件浏览")
    events_json = st.text_input("浏览数据源", value=defaults["events_json_path"], key="kb_events_browser_path")
    events = _read_json(events_json)
    if not isinstance(events, list):
        st.info("事件索引不可用，请先构建知识库。")
        return
    if not events:
        st.info("事件索引为空。")
        return

    levels = sorted({str(e.get("intensity_level", "unknown")) for e in events})
    seasons = sorted({str(e.get("season", "")) for e in events if str(e.get("season", ""))})
    c1, c2, c3 = st.columns(3)
    kw = c1.text_input("关键词（事件ID/名称）", value="", key="kb_event_kw")
    lv = c2.multiselect("强度筛选", options=levels, default=[], key="kb_event_level")
    sy = c3.multiselect("年份筛选", options=seasons, default=[], key="kb_event_season")

    filtered: list[dict[str, Any]] = []
    kw_lower = kw.strip().lower()
    for e in events:
        eid = str(e.get("event_id", ""))
        name = str(e.get("name", ""))
        level = str(e.get("intensity_level", "unknown"))
        season = str(e.get("season", ""))
        if kw_lower and kw_lower not in f"{eid} {name}".lower():
            continue
        if lv and level not in lv:
            continue
        if sy and season not in sy:
            continue
        filtered.append(e)

    st.caption(f"当前匹配 {len(filtered)} / {len(events)} 条事件")
    page_size = st.slider("每页展示", min_value=10, max_value=100, value=20, step=10, key="kb_page_size")
    max_page = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.number_input("页码", min_value=1, max_value=max_page, value=1, step=1, key="kb_page_no")
    start = (int(page) - 1) * int(page_size)
    end = min(len(filtered), start + int(page_size))
    show = filtered[start:end]

    rows = []
    for e in show:
        rows.append(
            {
                "事件ID": e.get("event_id", "-"),
                "名称": e.get("name", ""),
                "年份": e.get("season", ""),
                "强度": _friendly_level(str(e.get("intensity_level", "unknown"))),
                "起止时间": f"{e.get('start_time','')} ~ {e.get('end_time','')}",
                "中心点": f"({float(e.get('center_lon',0.0)):.2f}, {float(e.get('center_lat',0.0)):.2f})",
                "峰值风速(kt)": e.get("peak_wind_kt", ""),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if show:
        st.markdown("**事件详情**")
        selected_id = st.selectbox(
            "选择事件查看详情",
            options=[str(e.get("event_id", "-")) for e in show],
            index=0,
            key="kb_event_detail_select",
        )
        selected = next((e for e in show if str(e.get("event_id", "-")) == selected_id), None)
        if selected is not None:
            detail_lines = [
                f"- 事件ID：{selected.get('event_id', '-')}",
                f"- 名称：{selected.get('name', '')}",
                f"- 年份：{selected.get('season', '')}",
                f"- 强度：{_friendly_level(str(selected.get('intensity_level', 'unknown')))}",
                f"- 轨迹点数：{selected.get('n_points', 0)}",
                f"- 时间范围：{selected.get('start_time', '')} ~ {selected.get('end_time', '')}",
                (
                    "- 空间包围盒："
                    f"lon[{float(selected.get('lon_min', 0.0)):.2f}, {float(selected.get('lon_max', 0.0)):.2f}], "
                    f"lat[{float(selected.get('lat_min', 0.0)):.2f}, {float(selected.get('lat_max', 0.0)):.2f}]"
                ),
                (
                    "- 中心点："
                    f"({float(selected.get('center_lon', 0.0)):.2f}, {float(selected.get('center_lat', 0.0)):.2f})"
                ),
                f"- 峰值风速(kt)：{selected.get('peak_wind_kt', '')}",
                f"- 检索键：{', '.join(selected.get('retrieval_keys', []))}",
            ]
            if hasattr(st, "popover"):
                with st.popover("查看详情弹层", use_container_width=False):
                    st.markdown("\n".join(detail_lines))
                    st.caption("原始事件 JSON")
                    st.json(selected)
            else:
                with st.expander("查看详情（当前版本不支持弹层，已降级）", expanded=False):
                    st.markdown("\n".join(detail_lines))
                    st.caption("原始事件 JSON")
                    st.json(selected)
    with st.expander("当前页事件原始 JSON", expanded=False):
        st.json(show)


def _render_demo_cases(defaults: dict[str, Any]) -> None:
    st.subheader("联动案例")
    demo_cases = _read_json(defaults["demo_cases_path"])
    if not isinstance(demo_cases, list):
        st.info("未找到 demo_cases.json，可先运行 scripts/demo_typhoon_kb_cases.py")
        return
    st.caption(f"已加载 {len(demo_cases)} 个预置案例")
    for case in demo_cases:
        cid = case.get("case_id", "unknown")
        with st.expander(f"{cid}", expanded=False):
            q = case.get("query", {})
            st.write(
                f"时间窗: {q.get('start_time','')} ~ {q.get('end_time','')} | "
                f"区域: lon[{q.get('lon_min','')},{q.get('lon_max','')}], "
                f"lat[{q.get('lat_min','')},{q.get('lat_max','')}]"
            )
            rows = case.get("results", [])
            if isinstance(rows, list) and rows:
                st.dataframe(_format_rows_for_user(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("该案例无命中事件。")
            with st.expander("案例原始 JSON", expanded=False):
                st.json(case)


def render() -> None:
    st.title("台风知识库")
    st.caption("面向演示与日常使用：快速检索、历史浏览、案例复现三合一；并作为系统口径与命题任务 (5) 的定调说明入口。")
    with st.expander("系统定调 · 与命题任务 (5) 对齐", expanded=False):
        st.markdown(
            "- **风-浪异常**：以结构化 `run_detect` 输出（含 3σ 等级、残差与可选 `assessment_note`）为报告基础；"
            "台风候选来自本地 `events.json` 时空检索 + DTW，非大模型臆测。\n"
            "- **历史台风**：检索范围由查询时间窗与海区决定，索引可来自 IBTrACS 等构建脚本；扩大窗区见 `app/config/demo.yaml` 的 `typhoon_link`。\n"
            "- **演示与实测**：上传视频路径若无水文残差，可能使用 peak_score 代理；配套 NPZ 含 `demo_wind_*` 时走演示风浪序列。"
        )
    defaults = _default_query_params()
    _render_kb_status(defaults["events_json_path"])
    t1, t2, t3 = st.tabs(["快速检索", "历史事件浏览", "联动案例"])
    with t1:
        _render_query_panel(defaults)
    with t2:
        _render_event_browser(defaults)
    with t3:
        _render_demo_cases(defaults)
