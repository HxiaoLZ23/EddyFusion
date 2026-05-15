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

from web_api.routers import eddy_dual, health, hydro_heatmap, offline_nc, realtime_nc, windwave_report

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
app.include_router(realtime_nc.router, prefix="/api")
app.include_router(hydro_heatmap.router, prefix="/api")
app.include_router(eddy_dual.router, prefix="/api")
app.include_router(windwave_report.router, prefix="/api")
