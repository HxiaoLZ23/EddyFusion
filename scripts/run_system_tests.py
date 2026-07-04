#!/usr/bin/env python3
"""Run local system tests and export thesis-ready test artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "submission" / "tables"
FIGURE_DIR = ROOT / "submission" / "figures"


@dataclass
class TestCaseResult:
    suite: str
    name: str
    status: str
    seconds: float
    message: str = ""


def _run_command(cmd: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )


def _parse_junit(path: Path) -> list[TestCaseResult]:
    if not path.is_file():
        return [
            TestCaseResult(
                suite="pytest",
                name="junit_xml_missing",
                status="failed",
                seconds=0.0,
                message=f"JUnit XML not found: {path}",
            )
        ]
    root = ET.parse(path).getroot()
    rows: list[TestCaseResult] = []
    for case in root.iter("testcase"):
        suite = str(case.attrib.get("classname", "pytest")).replace("tests.", "")
        name = str(case.attrib.get("name", "unknown"))
        seconds = float(case.attrib.get("time", "0") or 0.0)
        status = "passed"
        message = ""
        failure = next(iter(case.findall("failure")), None)
        error = next(iter(case.findall("error")), None)
        skipped = next(iter(case.findall("skipped")), None)
        if failure is not None:
            status = "failed"
            message = str(failure.attrib.get("message", "") or (failure.text or "")).strip()
        elif error is not None:
            status = "error"
            message = str(error.attrib.get("message", "") or (error.text or "")).strip()
        elif skipped is not None:
            status = "skipped"
            message = str(skipped.attrib.get("message", "") or (skipped.text or "")).strip()
        rows.append(TestCaseResult(suite=suite, name=name, status=status, seconds=seconds, message=message))
    return rows


def _status_cn(status: str) -> str:
    return {
        "passed": "通过",
        "failed": "失败",
        "error": "错误",
        "skipped": "跳过",
    }.get(status, status)


# 论文表 6-7：编号、测试项、接口/方法、预期结果
_PAPER_T_MAP: dict[str, tuple[str, str, str, str]] = {
    "test_T1_preprocess_meta_probe": (
        "T1",
        "NC 元数据探测",
        "`GET /api/preprocess/meta`",
        "返回 `time_len`、`variables`、`variable_map`（含 `eddy_ready`）",
    ),
    "test_T2_preprocess_subset_roi": (
        "T2",
        "时空裁剪（ROI + 时间索引）",
        "`POST /api/preprocess/subset`",
        "子集 NC 写入 `app/data/nc_uploads/subsets/`",
    ),
    "test_T3_eddy_preview_frame": (
        "T3",
        "涡旋单帧预览",
        "`POST /api/eddy/preview-frame`",
        "PNG `data URL` + `stats_rows`（YOLO 或 ADT 降级）",
    ),
    "test_T4_eddy_dual_mp4_staged_job": (
        "T4",
        "涡旋双路 MP4（异步分阶段）",
        "`POST /api/jobs`（`eddy_dual_mp4`）",
        "job 至 `done`；底图路与标注路 MP4 可访问",
    ),
    "test_T5_windwave_forecast": (
        "T5",
        "风浪预测（同步）",
        "`POST /api/windwave/forecast`",
        "`series`、`anomaly_segments`、`typhoon_candidates`、异常等级",
    ),
    "test_T6_report_save_and_history": (
        "T6",
        "结构化报告归档",
        "`POST /api/report/structured` → `save` → `history`",
        "可列表、按 id 读取 Markdown 正文",
    ),
    "test_T7_async_job_windwave_forecast": (
        "T7",
        "风浪预测（异步 job）",
        "`POST /api/jobs`（`windwave_forecast`）",
        "轮询至 `done`，`result.series` 非空",
    ),
    "test_T8_realtime_connector_status": (
        "T8",
        "准实时连接器状态",
        "`GET /api/realtime/status`",
        "`connected`、`poll_dir`、`source` 等字段完整",
    ),
}


def _write_paper_t_table(rows: list[TestCaseResult], path: Path, *, table_title: str) -> None:
    """论文 §6.3 表 6-7：系统功能测试结果（T1～T8）。"""
    by_name = {r.name: r for r in rows}
    lines = [
        table_title,
        "",
        "> 由 `python scripts/run_system_tests.py --skip-ui` 自动生成；用例实现见 `tests/test_paper_system_api.py`。",
        "> 测试数据为仓库内合成小 NC（`demo_eddy_nc` / `demo_windwave_nc` fixture），验证 FastAPI 演示链路。",
        "",
        "**表 6-7  系统功能测试结果**",
        "",
        "| 编号 | 测试项 | 接口/方法 | 预期结果 | 测试结果 | 耗时(s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for test_name, (code, title, api, expect) in _PAPER_T_MAP.items():
        row = by_name.get(test_name)
        if row is None:
            status = "未执行"
            secs = "—"
        else:
            status = _status_cn(row.status)
            secs = f"{row.seconds:.3f}"
        lines.append(f"| {code} | {title} | {api} | {expect} | {status} | {secs} |")
    passed = sum(1 for n in _PAPER_T_MAP if by_name.get(n) and by_name[n].status == "passed")
    skipped = sum(1 for n in _PAPER_T_MAP if by_name.get(n) and by_name[n].status == "skipped")
    total = len(_PAPER_T_MAP)
    lines.extend(
        [
            "",
            f"**汇总**：{passed} 通过 / {skipped} 跳过 / {total} 项（T4 无本地 3ch 权重时跳过，不影响其余项）。",
            "",
            "复现命令：",
            "",
            "```powershell",
            "python -m pytest tests/test_paper_system_api.py -v",
            "# 或",
            "python scripts/run_system_tests.py --skip-ui",
            "```",
            "",
            "说明：本表为 **API 自动化**口径；前端页面（监测总览上传、报告管理 LLM 解读等）见 §5 界面截图与答辩演示脚本。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(rows: list[TestCaseResult], path: Path) -> None:
    lines = [
        "# 系统功能测试结果表",
        "",
        "| 测试层级 | 测试用例 | 测试结果 | 耗时(s) | 说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        message = row.message.replace("\n", " ").replace("|", "/")
        if len(message) > 120:
            message = message[:117] + "..."
        lines.append(
            f"| {row.suite} | `{row.name}` | {_status_cn(row.status)} | {row.seconds:.3f} | {message or '-'} |"
        )
    lines.extend(
        [
            "",
            "说明：本表由 `python scripts/run_system_tests.py` 自动生成，测试数据采用本地 demo/合成小数据，"
            "用于验证系统核心功能链路是否可在本地稳定运行。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_overview_figure(rows: list[TestCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = {
        "passed": sum(1 for r in rows if r.status == "passed"),
        "failed": sum(1 for r in rows if r.status == "failed"),
        "error": sum(1 for r in rows if r.status == "error"),
        "skipped": sum(1 for r in rows if r.status == "skipped"),
    }
    labels = ["passed", "failed", "error", "skipped"]
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    bars = ax.bar(labels, values, color=["#2ca02c", "#d62728", "#ff7f0e", "#7f7f7f"])
    ax.set_title("Local System Test Overview")
    ax.set_ylabel("Test cases")
    ax.set_ylim(0, max(values + [1]) + 1)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _load_ui_rows(path: Path) -> list[TestCaseResult]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [TestCaseResult("ui", "ui_summary_parse", "error", 0.0, str(exc))]
    rows = []
    for item in payload.get("screenshots", []):
        status = "passed" if item.get("status") == "passed" else "failed"
        rows.append(
            TestCaseResult(
                suite="ui",
                name=str(item.get("name", "screenshot")),
                status=status,
                seconds=float(item.get("seconds", 0.0) or 0.0),
                message=str(item.get("path") or item.get("message") or ""),
            )
        )
    if payload.get("status") == "failed" and not rows:
        rows.append(TestCaseResult("ui", "capture_system_ui", "failed", 0.0, str(payload.get("message", ""))))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ui", action="store_true", help="only run pytest/API tests")
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    junit_path = TABLE_DIR / "system_test_junit.xml"
    ui_json_path = TABLE_DIR / "system_ui_capture_summary.json"
    started = time.perf_counter()

    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests",
        f"--junitxml={junit_path.as_posix()}",
    ]
    pytest_result = _run_command(pytest_cmd)
    pytest_log = TABLE_DIR / "system_pytest_output.log"
    pytest_log.write_text(pytest_result.stdout, encoding="utf-8")
    rows = _parse_junit(junit_path)

    if not args.skip_ui:
        capture_cmd = [
            sys.executable,
            "scripts/capture_system_ui.py",
            "--output-json",
            ui_json_path.as_posix(),
        ]
        capture_result = _run_command(capture_cmd)
        (TABLE_DIR / "system_ui_capture_output.log").write_text(capture_result.stdout, encoding="utf-8")
        rows.extend(_load_ui_rows(ui_json_path))

    summary = {
        "status": "passed" if rows and all(r.status in {"passed", "skipped"} for r in rows) else "failed",
        "generated_at": int(time.time()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "total": len(rows),
        "passed": sum(1 for r in rows if r.status == "passed"),
        "failed": sum(1 for r in rows if r.status == "failed"),
        "error": sum(1 for r in rows if r.status == "error"),
        "skipped": sum(1 for r in rows if r.status == "skipped"),
        "pytest_exit_code": pytest_result.returncode,
        "artifacts": {
            "functional_table": "submission/tables/system_functional_test_results.md",
            "paper_t1_t8_table": "submission/tables/paper_system_test_t1_t8.md",
            "table_6_7": "submission/tables/table_6_7_system_function_test.md",
            "summary_json": "submission/tables/system_test_summary.json",
            "overview_figure": "submission/figures/system_test_overview.png",
            "pytest_log": "submission/tables/system_pytest_output.log",
            "ui_summary": "submission/tables/system_ui_capture_summary.json",
        },
    }

    _write_markdown(rows, TABLE_DIR / "system_functional_test_results.md")
    _write_paper_t_table(
        rows,
        TABLE_DIR / "table_6_7_system_function_test.md",
        table_title="# 表 6-7  系统功能测试结果",
    )
    _write_paper_t_table(
        rows,
        TABLE_DIR / "paper_system_test_t1_t8.md",
        table_title="# 论文系统功能测试表（§6.3 T1～T8，同表 6-7）",
    )
    (TABLE_DIR / "system_test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_overview_figure(rows, FIGURE_DIR / "system_test_overview.png")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
