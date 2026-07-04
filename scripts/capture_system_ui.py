#!/usr/bin/env python3
"""Capture browser screenshots for thesis paper figures (§4–§5 React+FastAPI UI)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "submission" / "figures"
TABLE_DIR = ROOT / "submission" / "tables"
API_URL = "http://127.0.0.1:8000/api/health"
WEB_URL = "http://127.0.0.1:5173"

PAPER_FIGURES: list[dict[str, str]] = [
    {
        "id": "fig4-4",
        "caption": "图4-4 系统主要界面原型图",
        "name": "fig4-4_system_main_prototype",
        "file": "fig4-4_system_main_prototype.png",
    },
    {
        "id": "fig5-10",
        "caption": "图5-10 系统主界面运行效果图",
        "name": "fig5-10_system_main_runtime",
        "file": "fig5-10_system_main_runtime.png",
    },
    {
        "id": "fig5-11",
        "caption": "图5-11 海洋环境场可视化界面",
        "name": "fig5-11_ocean_field_visualization",
        "file": "fig5-11_ocean_field_visualization.png",
    },
    {
        "id": "fig5-12",
        "caption": "图5-12 涡旋识别结果展示界面",
        "name": "fig5-12_eddy_detection_result",
        "file": "fig5-12_eddy_detection_result.png",
    },
    {
        "id": "fig5-13",
        "caption": "图5-13 风浪预测与异常预警展示界面",
        "name": "fig5-13_windwave_forecast_warning",
        "file": "fig5-13_windwave_forecast_warning.png",
    },
    {
        "id": "fig5-14",
        "caption": "图5-14 分析报告导出界面",
        "name": "fig5-14_report_export",
        "file": "fig5-14_report_export.png",
    },
]

LEGACY_PAGES = [
    {
        "name": "system_offline_dashboard",
        "path_url": "/offline",
        "path": FIGURE_DIR / "system_offline_dashboard.png",
        "expected": "离线系统",
    },
    {
        "name": "system_eddy_panel",
        "path_url": "/offline/l1/eddy",
        "path": FIGURE_DIR / "system_eddy_panel.png",
        "expected": "涡旋",
    },
    {
        "name": "system_windwave_panel",
        "path_url": "/offline/l1/windwave",
        "path": FIGURE_DIR / "system_windwave_panel.png",
        "expected": "风浪",
    },
    {
        "name": "system_typhoon_kb_panel",
        "path_url": "/offline/typhoon-kb",
        "path": FIGURE_DIR / "system_typhoon_kb_panel.png",
        "expected": "台风",
    },
]


def _ensure_demo_assets() -> Path:
    subprocess.run([sys.executable, "scripts/seed_typhoon_kb_demo.py"], cwd=str(ROOT), check=False)
    subprocess.run([sys.executable, "scripts/generate_demo_nc_three_modules.py"], cwd=str(ROOT), check=True)
    demo = ROOT / "outputs" / "demo_nc_three_modules" / "mod_fused_stream_windwave_video.nc"
    if not demo.is_file():
        demo = ROOT / "outputs" / "demo_nc_three_modules" / "mod2_ocean_wind_wave.nc"
    if not demo.is_file():
        raise RuntimeError("demo NetCDF was not generated")
    return demo


@dataclass
class ManagedProcess:
    name: str
    proc: subprocess.Popen[str] | None

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _url_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 500
    except (OSError, urllib.error.URLError):
        return False


def _wait_url(url: str, *, seconds: float = 45.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if _url_ok(url):
            return True
        time.sleep(0.8)
    return False


def _popen(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", errors="replace")
    return subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _npm_cmd() -> list[str]:
    if os.name == "nt":
        return ["cmd", "/c", "npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"]
    return ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"]


def _start_api_if_needed() -> ManagedProcess:
    if _url_ok(API_URL):
        return ManagedProcess("api", None)
    proc = _popen(
        [sys.executable, "-m", "uvicorn", "web_api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT,
        log_path=TABLE_DIR / "system_ui_api_server.log",
    )
    if not _wait_url(API_URL):
        raise RuntimeError("FastAPI server did not become ready on http://127.0.0.1:8000")
    return ManagedProcess("api", proc)


def _start_web_if_needed() -> ManagedProcess:
    if _url_ok(WEB_URL):
        return ManagedProcess("web", None)
    proc = _popen(
        _npm_cmd(),
        cwd=ROOT / "web",
        log_path=TABLE_DIR / "system_ui_web_server.log",
    )
    if not _wait_url(WEB_URL):
        raise RuntimeError("Vite web server did not become ready on http://127.0.0.1:5173")
    return ManagedProcess("web", proc)


def _write_failure(output_json: Path, message: str) -> int:
    payload = {
        "status": "failed",
        "generated_at": int(time.time()),
        "message": message,
        "screenshots": [],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(message)
    return 1


def _append_shot(
    rows: list[dict[str, Any]],
    *,
    name: str,
    caption: str,
    path: Path,
    started: float,
    status: str = "passed",
    message: str | None = None,
) -> None:
    row: dict[str, Any] = {
        "name": name,
        "caption": caption,
        "status": status,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "seconds": round(time.perf_counter() - started, 3),
    }
    if message:
        row["message"] = message
    rows.append(row)


def _launch_browser(p):
    launch_errors: list[str] = []
    browser = None
    for kwargs in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            browser = p.chromium.launch(headless=True, **kwargs)
            break
        except Exception as exc:
            launch_errors.append(str(exc).splitlines()[0])
    if browser is None:
        raise RuntimeError(
            "无法启动 Chromium/Chrome/Edge；可执行 `python -m playwright install chromium` 后重试。"
            f" 原因: {'; '.join(launch_errors)}"
        )
    return browser


def _nav_click(page, label: str) -> None:
    page.locator("header nav").get_by_role("link", name=label, exact=True).click()


def _wait_body_text(page, text: str, *, timeout_ms: int = 60_000) -> None:
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        try:
            body = page.locator("body").inner_text(timeout=3000)
            if text in body:
                return
        except Exception:
            pass
        page.wait_for_timeout(800)
    raise RuntimeError(f"expected text not found within {timeout_ms}ms: {text}")


def _upload_demo_nc(page, demo_nc: Path) -> None:
    upload = page.locator('input[type="file"]').first
    upload.wait_for(state="attached", timeout=15_000)
    upload.set_input_files(str(demo_nc))


def _write_paper_index(rows: list[dict[str, Any]]) -> None:
    index_path = FIGURE_DIR / "paper_figures_index.md"
    lines = [
        "# 论文系统配图索引（React + FastAPI）",
        "",
        "| 图号 | 说明 | 文件 | 状态 |",
        "|------|------|------|------|",
    ]
    by_name = {str(r["name"]): r for r in rows}
    for spec in PAPER_FIGURES:
        row = by_name.get(spec["name"], {})
        status = row.get("status", "skipped")
        rel = row.get("path", spec["file"])
        lines.append(f"| {spec['id']} | {spec['caption']} | `{rel}` | {status} |")
    lines.extend(
        [
            "",
            "生成命令：`python scripts/capture_system_ui.py`",
            "",
        ]
    )
    index_path.write_text("\n".join(lines), encoding="utf-8")


def _capture_paper_figures(demo_nc: Path) -> list[dict[str, Any]]:
    from playwright.sync_api import sync_playwright

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    screenshots: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser = _launch_browser(p)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)

            # --- 图4-4：监测总览原型（未上传） ---
            t0 = time.perf_counter()
            spec = PAPER_FIGURES[0]
            out = FIGURE_DIR / spec["file"]
            page.goto(f"{WEB_URL}/monitor", wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(1500)
            _wait_body_text(page, "监测总览", timeout_ms=20_000)
            _wait_body_text(page, "任务配置", timeout_ms=20_000)
            page.screenshot(path=str(out), full_page=True)
            _append_shot(screenshots, name=spec["name"], caption=spec["caption"], path=out, started=t0)

            # 上传 demo NC（同浏览器上下文内勿 page.goto 整页刷新，否则会话 state 会丢失）
            _upload_demo_nc(page, demo_nc)
            _wait_body_text(page, "已接入 NetCDF", timeout_ms=120_000)
            page.wait_for_timeout(12_000)

            # --- 图5-10：监测总览运行态（停留当前页） ---
            t0 = time.perf_counter()
            spec = PAPER_FIGURES[1]
            out = FIGURE_DIR / spec["file"]
            page.screenshot(path=str(out), full_page=True)
            _append_shot(screenshots, name=spec["name"], caption=spec["caption"], path=out, started=t0)

            # --- 涡旋分析页（SPA 导航） ---
            _nav_click(page, "涡旋分析")
            page.wait_for_timeout(2000)
            preview_btn = page.get_by_role("button", name="加载预览帧")
            preview_btn.click()
            page.locator('img[alt="eddy preview frame"]').wait_for(state="visible", timeout=180_000)
            page.wait_for_timeout(1200)

            # --- 图5-11：中央环境场卡片 ---
            t0 = time.perf_counter()
            spec = PAPER_FIGURES[2]
            out = FIGURE_DIR / spec["file"]
            field_card = page.locator("strong").filter(has_text="中央场图").locator("xpath=..")
            field_card.screenshot(path=str(out))
            _append_shot(screenshots, name=spec["name"], caption=spec["caption"], path=out, started=t0)

            # --- 图5-12：涡旋识别全页 ---
            t0 = time.perf_counter()
            spec = PAPER_FIGURES[3]
            out = FIGURE_DIR / spec["file"]
            page.screenshot(path=str(out), full_page=True)
            _append_shot(screenshots, name=spec["name"], caption=spec["caption"], path=out, started=t0)

            # --- 风浪分析页（SPA 导航） ---
            _nav_click(page, "风浪分析")
            page.wait_for_timeout(2000)
            page.get_by_role("button", name="运行风浪预测").click()
            page.locator('svg[aria-label="风浪预测与异常高亮"]').wait_for(state="visible", timeout=240_000)
            page.wait_for_timeout(1500)

            # --- 图5-13：风浪预测与异常预警 ---
            t0 = time.perf_counter()
            spec = PAPER_FIGURES[4]
            out = FIGURE_DIR / spec["file"]
            page.screenshot(path=str(out), full_page=True)
            _append_shot(screenshots, name=spec["name"], caption=spec["caption"], path=out, started=t0)

            # 导出报告并进入报告管理
            page.get_by_role("button", name="导出结构化报告").click()
            page.wait_for_timeout(12_000)

            _nav_click(page, "报告管理")
            page.wait_for_timeout(3000)
            _wait_body_text(page, "报告管理", timeout_ms=20_000)

            # --- 图5-14：分析报告导出界面 ---
            t0 = time.perf_counter()
            spec = PAPER_FIGURES[5]
            out = FIGURE_DIR / spec["file"]
            try:
                page.wait_for_timeout(3000)
                if page.locator("button").filter(has_text="风浪异常报告").count() == 0:
                    first_report = page.locator("button").filter(has_text=".nc").first
                    if first_report.count() > 0:
                        first_report.click()
                        page.wait_for_timeout(1500)
                page.screenshot(path=str(out), full_page=True)
                _append_shot(screenshots, name=spec["name"], caption=spec["caption"], path=out, started=t0)
            except Exception as exc:
                page.screenshot(path=str(out), full_page=True)
                _append_shot(
                    screenshots,
                    name=spec["name"],
                    caption=spec["caption"],
                    path=out,
                    started=t0,
                    status="failed",
                    message=str(exc),
                )
        finally:
            browser.close()

    return screenshots


def _capture_legacy(demo_nc: Path) -> list[dict[str, Any]]:
    from playwright.sync_api import sync_playwright

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    screenshots: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser = _launch_browser(p)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)
            page.goto(f"{WEB_URL}/offline", wait_until="domcontentloaded", timeout=20_000)
            page.wait_for_timeout(1200)
            upload = page.locator('input[type="file"]').first
            if upload.count() > 0:
                upload.set_input_files(str(demo_nc))
                page.wait_for_timeout(10_000)
            for item in LEGACY_PAGES:
                started = time.perf_counter()
                try:
                    page.evaluate(
                        """(path) => {
                            window.history.pushState({}, '', path);
                            window.dispatchEvent(new PopStateEvent('popstate'));
                        }""",
                        item["path_url"],
                    )
                    page.wait_for_timeout(3000)
                    if item["name"] == "system_windwave_panel":
                        page.wait_for_timeout(3000)
                    text = page.locator("body").inner_text(timeout=5000)
                    expected = str(item["expected"])
                    if expected not in text:
                        raise RuntimeError(f"expected text not found: {expected}")
                    page.screenshot(path=str(item["path"]), full_page=True)
                    _append_shot(
                        screenshots,
                        name=str(item["name"]),
                        caption=str(item["name"]),
                        path=Path(item["path"]),
                        started=started,
                    )
                except Exception as exc:
                    _append_shot(
                        screenshots,
                        name=str(item["name"]),
                        caption=str(item["name"]),
                        path=Path(item["path"]),
                        started=started,
                        status="failed",
                        message=str(exc),
                    )
        finally:
            browser.close()

    return screenshots


def _capture(output_json: Path, *, legacy: bool) -> int:
    demo_nc = _ensure_demo_assets()
    screenshots = _capture_paper_figures(demo_nc)
    if legacy:
        screenshots.extend(_capture_legacy(demo_nc))
    _write_paper_index(screenshots)

    status = "passed" if all(row["status"] == "passed" for row in screenshots) else "failed"
    payload = {
        "status": status,
        "generated_at": int(time.time()),
        "paper_figures": [s["id"] for s in PAPER_FIGURES],
        "screenshots": screenshots,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=str(TABLE_DIR / "system_ui_capture_summary.json"))
    parser.add_argument("--legacy", action="store_true", help="Also capture old /offline route screenshots")
    args = parser.parse_args()
    output_json = (ROOT / args.output_json).resolve() if not Path(args.output_json).is_absolute() else Path(args.output_json)

    processes: list[ManagedProcess] = []
    try:
        try:
            processes.append(_start_api_if_needed())
            processes.append(_start_web_if_needed())
        except Exception as exc:
            return _write_failure(output_json, str(exc))
        try:
            return _capture(output_json, legacy=args.legacy)
        except ImportError as exc:
            return _write_failure(output_json, f"Playwright is not installed: {exc}")
        except Exception as exc:
            return _write_failure(output_json, str(exc))
    finally:
        for proc in reversed(processes):
            proc.stop()


if __name__ == "__main__":
    raise SystemExit(main())
