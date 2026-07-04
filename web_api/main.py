from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
for _p in (_APP, _ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web_api.routers import (
    eddy_dual,
    health,
    hydro_heatmap,
    jobs,
    offline_nc,
    preprocess,
    realtime_nc,
    report,
    typhoon_kb,
    windwave_llm,
    windwave_report,
)

try:
    from src.utils.dashscope_settings import apply_dashscope_env_from_file

    apply_dashscope_env_from_file()
except Exception:
    pass

app = FastAPI(title="EddyFusion Ocean API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(offline_nc.router, prefix="/api")
app.include_router(preprocess.router, prefix="/api")
app.include_router(realtime_nc.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(hydro_heatmap.router, prefix="/api")
app.include_router(eddy_dual.router, prefix="/api")
app.include_router(windwave_report.router, prefix="/api")
app.include_router(windwave_llm.router, prefix="/api")
app.include_router(report.router, prefix="/api")
app.include_router(typhoon_kb.router, prefix="/api")


@app.on_event("startup")
def _ensure_typhoon_kb_demo() -> None:
    """仅当 data/processed/anomaly/typhoon_kb/events.json 不存在时写入演示索引（不覆盖已有库）。"""
    events = _ROOT / "data/processed/anomaly/typhoon_kb/events.json"
    if events.is_file():
        return
    try:
        import subprocess

        subprocess.run(
            [sys.executable, str(_ROOT / "scripts/seed_typhoon_kb_demo.py")],
            cwd=str(_ROOT),
            check=False,
            timeout=30,
        )
    except Exception:
        pass
