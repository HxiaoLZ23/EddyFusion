#!/usr/bin/env python3
"""
Smoke test：验证百炼「模型部署」是否能通过 DashScope 正常作答。

使用前（同一 PowerShell 会话）：
  $env:DASHSCOPE_API_KEY = "<你的密钥>"

重要：`Generation.call` 的 `--model` 必须是控制台返回的 **`deployed_model` 代码**
（形如 `qwen-plus-xxxx-sample01`），**不是**你给部署起的别名（例如 `qwen3-14b_typhoon`）。

先列出可用的部署代号：
  python scripts/test_dashscope_model.py --list-deployments

再用其中一条 RUNNING 的 `deployed_model` 调用：
  python scripts/test_dashscope_model.py --model <deployed_model>

可选环境变量：`DASHSCOPE_MODEL`、`DASHSCOPE_WORKSPACE`（控制台若要求 workspace）

风浪异常烟测（模拟 run_detect → LLM 输入）示例：
  python scripts/test_dashscope_model.py --scenario no_typhoon
  python scripts/test_dashscope_model.py --scenario with_typhoon
  python scripts/test_dashscope_model.py --scenario degrade
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.llm_sft_report_prompt import SYSTEM_CHATML_ANOMALY_REPORT as SYSTEM_ANOMALY_REPORT


def _validate_report_json(text: str) -> tuple[bool, str]:
    """校验 assistant 是否为完整四段 JSON。"""
    t = (text or "").strip()
    if not t:
        return False, "empty"
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        return False, f"json_decode: {e}"
    for k in ("summary_anomaly", "impact", "historical_analogy", "actions"):
        if k not in obj:
            return False, f"missing_key:{k}"
    acts = obj.get("actions")
    if not isinstance(acts, list) or len(acts) < 3:
        return False, "actions_not_list_or_too_short"
    return True, "ok"


def _sample_detect_payload(scenario: str) -> dict:
    """贴近 SFT / 运行时裁剪后的结构化输入（演示用虚构数值）。"""
    base_anomaly = {
        "anomaly_level": "high",
        "anomaly_index": 0.71,
        "wind_residual": 1.82,
        "wind_z": 3.35,
        "wave_residual": 0.58,
        "wave_z": 2.88,
        "threshold_rule": "3sigma",
    }
    query = {
        "start_time": "2024-09-08 06:00:00",
        "end_time": "2024-09-09 18:00:00",
        "lon_min": 120.5,
        "lon_max": 125.8,
        "lat_min": 28.2,
        "lat_max": 32.6,
    }

    if scenario == "with_typhoon":
        return {
            "anomaly_result": {**base_anomaly, "assessment_note": ""},
            "typhoon_link": {
                "query": query,
                "candidates": [
                    {
                        "event_id": "2024146N08125",
                        "name": "DEMO_CYCLONE_SAMPLE",
                        "start_time": "2024-09-07 00:00:00",
                        "end_time": "2024-09-11 06:00:00",
                        "intensity_level": "TY",
                        "peak_wind_kt": 75.0,
                        "bbox_overlap_ratio": 0.42,
                        "time_overlap_hours": 28.5,
                        "score": 86.32,
                        "dtw_distance": 1.28,
                    }
                ],
            },
        }

    if scenario == "degrade":
        deg = dict(base_anomaly)
        deg["anomaly_level"] = "medium"
        deg["assessment_note"] = "风速序列可用于 DTW；波高观测曲线缺失，已启用无曲线降级。"
        deg["wave_z"] = None
        return {
            "anomaly_result": deg,
            "typhoon_link": {
                "query": query,
                "candidates": [
                    {
                        "event_id": "2024180N11130",
                        "name": "DEMO_LOW_CONF",
                        "start_time": "2024-09-09 06:00:00",
                        "end_time": "2024-09-10 00:00:00",
                        "score": 22.01,
                        "dtw_distance": None,
                    }
                ],
            },
        }

    # no_typhoon：知识库候选为空（演示常见痛点）
    return {
        "anomaly_result": {**base_anomaly, "assessment_note": ""},
        "typhoon_link": {"query": query, "candidates": []},
    }


def _print_response(resp: object) -> None:
    code = getattr(resp, "status_code", None) or getattr(resp, "status", None)
    print("status_code:", code)
    if code not in (200, "200", None):
        print("code:", getattr(resp, "code", ""))
        print("message:", getattr(resp, "message", ""))
        msg = str(getattr(resp, "message", "") or "")
        if "not exist" in msg.lower() or "Model not exist" in msg:
            print(
                "\n提示：百炼部署调用需使用「部署列表」里的 deployed_model 字符串，"
                "不是部署显示名称。请运行：\n"
                "  python scripts/test_dashscope_model.py --list-deployments",
                file=sys.stderr,
            )
        return
    out = getattr(resp, "output", None)
    if out is None:
        print("output:", resp)
        return
    # dict-like
    if isinstance(out, dict):
        choices = out.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            content = msg.get("content") or choices[0].get("text")
            print("reply:\n", content)
            ok, why = _validate_report_json(str(content or ""))
            print("json_check:", ok, why)
            return
        print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return
    # object with choices
    ch = getattr(out, "choices", None)
    if ch:
        first = ch[0]
        if isinstance(first, dict):
            msg = first.get("message", {})
            print("reply:\n", msg.get("content", first))
            ok, why = _validate_report_json(str(msg.get("content") or ""))
            print("json_check:", ok, why)
            return
        print("reply:\n", getattr(first, "content", first))
        ok, why = _validate_report_json(str(getattr(first, "content", "") or ""))
        print("json_check:", ok, why)
        return
    print("output repr:", repr(out)[:2000])


def _list_deployments(api_key: str) -> None:
    url = (
        "https://dashscope.aliyuncs.com/api/v1/deployments"
        "?page_no=1&page_size=100"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {err[:2000]}", file=sys.stderr)
        raise SystemExit(1) from e

    out = body.get("output") or {}
    deployments = out.get("deployments") or []
    if not deployments:
        print("未发现部署任务（output.deployments 为空）。请先在百炼控制台完成部署且状态为 RUNNING。")
        print(json.dumps(body, ensure_ascii=False, indent=2)[:4000])
        return
    print("以下字段中的 deployed_model 即调用时应填入的 --model：\n")
    for d in deployments:
        print(
            f"- deployed_model: {d.get('deployed_model')}\n"
            f"  model_name: {d.get('model_name')}  status: {d.get('status')}\n"
            f"  workspace_id: {d.get('workspace_id','')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="DashScope / 百炼模型烟测")
    parser.add_argument(
        "--list-deployments",
        action="store_true",
        help="列出专属部署代号（deployed_model），用于填 --model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("DASHSCOPE_MODEL") or "",
        help="必须使用部署接口返回的 deployed_model，不是控制台显示别名",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=("no_typhoon", "with_typhoon", "degrade"),
        default="no_typhoon",
        help="风浪模块烟测预设：无候选台风 / 有候选 / 降级+弱候选",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="若不空，则覆盖 --scenario：原始 user content（仍带 system）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("DASHSCOPE_MAX_TOKENS", "2048")),
        help="避免 assistant JSON 在 actions 处被截断",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key or api_key == "你的新key":
        print(
            "未设置有效的 DASHSCOPE_API_KEY。\n"
            "请在同一 PowerShell 窗口执行：\n"
            '  $env:DASHSCOPE_API_KEY = "<真实密钥>"\n'
            "然后再运行：\n"
            "  python scripts/test_dashscope_model.py\n",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.list_deployments:
        _list_deployments(api_key)
        return

    model = (args.model or "").strip()
    if not model:
        print(
            "请指定模型部署代号。\n"
            "1) 列出可用部署：  python scripts/test_dashscope_model.py --list-deployments\n"
            "2) 使用其中 RUNNING 的 deployed_model：  --model <deployed_model>\n"
            "或设置环境变量：  $env:DASHSCOPE_MODEL = '<deployed_model>'\n",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        from dashscope import Generation
    except ImportError:
        print("请先安装：pip install dashscope>=1.19.0", file=sys.stderr)
        raise SystemExit(1)

    if (args.prompt or "").strip():
        user_content = (args.prompt or "").strip()
        scenario_note = "(自定义 --prompt)"
    else:
        payload = _sample_detect_payload(args.scenario)
        user_content = json.dumps(payload, ensure_ascii=False)
        scenario_note = f"(scenario={args.scenario})"

    messages = [
        {"role": "system", "content": SYSTEM_ANOMALY_REPORT},
        {"role": "user", "content": user_content},
    ]

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "api_key": api_key,
        "result_format": "message",
        "enable_thinking": False,
        "max_tokens": int(args.max_tokens),
    }
    ws = os.environ.get("DASHSCOPE_WORKSPACE", "").strip()
    if ws:
        kwargs["workspace"] = ws

    print("model:", kwargs["model"])
    print("scenario:", scenario_note)
    if len(user_content) > 800:
        print("user_payload:", user_content[:800] + " ...（已截断）")
    else:
        print("user_payload:", user_content)
    resp = Generation.call(**kwargs)
    _print_response(resp)


if __name__ == "__main__":
    main()
