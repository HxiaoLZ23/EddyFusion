from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from web_api.deps import REPO_ROOT

router = APIRouter(prefix="/typhoon-kb", tags=["typhoon-kb"])

DEFAULT_EVENTS_JSON = "data/processed/anomaly/typhoon_kb/events.json"
DEFAULT_DEMO_CASES = "data/processed/anomaly/typhoon_kb/demo_cases.json"


def _events_path(custom: str | None = None) -> Path:
    rel = custom or DEFAULT_EVENTS_JSON
    p = Path(rel)
    if not p.is_absolute():
        p = (REPO_ROOT / rel).resolve()
    return p


def _parse_time(raw: str) -> datetime:
    s = (raw or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {raw}")


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取 JSON 失败: {e}") from e
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="索引格式应为 JSON 数组")
    return [x for x in data if isinstance(x, dict)]


@router.get("/status")
def typhoon_kb_status() -> dict[str, Any]:
    p = _events_path()
    ready = p.is_file()
    count = 0
    source = None
    if ready:
        count = len(_load_json_list(p))
        idx = (REPO_ROOT / "data/processed/anomaly/typhoon_index.json").resolve()
        if idx.is_file():
            try:
                meta = json.loads(idx.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    source = meta.get("source")
                    if meta.get("events_count") is not None:
                        count = int(meta.get("events_count", count))
            except Exception:
                pass
    return {
        "ready": ready,
        "events_json_path": str(p),
        "events_count": count,
        "source": source,
        "seed_hint": "python scripts/seed_typhoon_kb_demo.py",
        "full_build_hint": "scripts/run_typhoon_kb.ps1",
    }


@router.get("/defaults")
def typhoon_kb_defaults() -> dict[str, Any]:
    from src.anomaly.eddy_typhoon_bridge import typhoon_query_bbox_from_configs
    from src.utils.config import load_yaml, resolve_path

    demo_cfg: dict[str, Any] = {}
    try:
        raw = load_yaml("app/config/demo.yaml")
        if isinstance(raw, dict):
            demo_cfg = raw
    except Exception:
        pass
    ty_cfg = demo_cfg.get("typhoon_link", {}) if isinstance(demo_cfg.get("typhoon_link"), dict) else {}

    end_dt = datetime.now()
    win_h = int(float(ty_cfg.get("default_window_hours", 240)))
    start_dt = end_dt - timedelta(hours=max(1, win_h))
    lon_min, lon_max, lat_min, lat_max = typhoon_query_bbox_from_configs()
    events_rel = str(ty_cfg.get("events_json_path", DEFAULT_EVENTS_JSON))
    demo_rel = str(ty_cfg.get("demo_cases_path", DEFAULT_DEMO_CASES))

    return {
        "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "top_k": int(float(ty_cfg.get("default_top_k", 5))),
        "events_json_path": events_rel,
        "demo_cases_path": demo_rel,
    }


class TyphoonQueryBody(BaseModel):
    start_time: str
    end_time: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    top_k: int = Field(default=10, ge=1, le=50)
    events_json_path: str | None = None


@router.post("/query")
def typhoon_kb_query(body: TyphoonQueryBody) -> dict[str, Any]:
    from src.anomaly.typhoon_kb import QueryBox, query_typhoon_events

    p = _events_path(body.events_json_path)
    if not p.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"台风事件索引不存在: {p}。可运行 python scripts/seed_typhoon_kb_demo.py",
        )
    try:
        q = QueryBox(
            start_time=_parse_time(body.start_time),
            end_time=_parse_time(body.end_time),
            lon_min=float(body.lon_min),
            lon_max=float(body.lon_max),
            lat_min=float(body.lat_min),
            lat_max=float(body.lat_max),
        )
        rows = query_typhoon_events(events_json_path=p, query=q, top_k=int(body.top_k))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {e}") from e

    return {
        "status": "success",
        "count": len(rows),
        "query": {
            "start_time": body.start_time,
            "end_time": body.end_time,
            "lon_min": body.lon_min,
            "lon_max": body.lon_max,
            "lat_min": body.lat_min,
            "lat_max": body.lat_max,
            "top_k": body.top_k,
        },
        "candidates": rows,
        "events_json_path": str(p),
    }


@router.get("/events")
def typhoon_kb_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=100),
    keyword: str = Query(""),
    level: str = Query(""),
    season: str = Query(""),
    events_json_path: str | None = None,
) -> dict[str, Any]:
    p = _events_path(events_json_path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"事件索引不存在: {p}")

    events = _load_json_list(p)
    kw = keyword.strip().lower()
    levels = {x.strip() for x in level.split(",") if x.strip()}
    seasons = {x.strip() for x in season.split(",") if x.strip()}

    filtered: list[dict[str, Any]] = []
    for e in events:
        eid = str(e.get("event_id", ""))
        name = str(e.get("name", ""))
        lv = str(e.get("intensity_level", "unknown"))
        sy = str(e.get("season", ""))
        if kw and kw not in f"{eid} {name}".lower():
            continue
        if levels and lv not in levels:
            continue
        if seasons and sy not in seasons:
            continue
        filtered.append(e)

    total = len(filtered)
    max_page = max(1, (total + page_size - 1) // page_size)
    page = min(page, max_page)
    start = (page - 1) * page_size
    end = min(total, start + page_size)

    all_levels = sorted({str(e.get("intensity_level", "unknown")) for e in events})
    all_seasons = sorted({str(e.get("season", "")) for e in events if str(e.get("season", ""))})

    return {
        "status": "success",
        "total": total,
        "page": page,
        "page_size": page_size,
        "max_page": max_page,
        "items": filtered[start:end],
        "facets": {"levels": all_levels, "seasons": all_seasons},
        "events_json_path": str(p),
    }


@router.get("/demo-cases")
def typhoon_kb_demo_cases(path: str | None = None) -> dict[str, Any]:
    rel = path or DEFAULT_DEMO_CASES
    p = Path(rel)
    if not p.is_absolute():
        p = (REPO_ROOT / rel).resolve()
    if not p.is_file():
        return {"status": "success", "cases": [], "path": str(p), "note": "未找到 demo_cases.json"}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    cases = data if isinstance(data, list) else []
    return {"status": "success", "cases": cases, "path": str(p)}
