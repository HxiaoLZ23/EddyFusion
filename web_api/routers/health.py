from __future__ import annotations

import os

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    out = {"status": "ok"}
    sha = os.environ.get("GIT_SHA", "").strip()
    if sha:
        out["git_sha"] = sha
    return out
