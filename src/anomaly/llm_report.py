"""风浪异常：将 run_detect 输出转为 LLM 用户 JSON，并可选调用 DashScope 生成四段解读报告。"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from src.anomaly.llm_sft_report_prompt import SYSTEM_CHATML_ANOMALY_REPORT

_ANOMALY_KEYS = (
    "anomaly_level",
    "anomaly_index",
    "wind_residual",
    "wind_z",
    "wave_residual",
    "wave_z",
    "threshold_rule",
    "assessment_note",
)
_CANDIDATE_KEYS = (
    "event_id",
    "name",
    "start_time",
    "end_time",
    "intensity_level",
    "peak_wind_kt",
    "score",
    "bbox_overlap_ratio",
    "time_overlap_hours",
    "dtw_distance",
)
_REPORT_KEYS = frozenset({"summary_anomaly", "impact", "historical_analogy", "actions"})


def build_user_payload_from_detect(detect_output: dict[str, Any]) -> dict[str, Any]:
    """裁剪为与 SFT 一致的 user JSON（不含 meta 调试键）。"""
    ar_in = detect_output.get("anomaly_result") if isinstance(detect_output, dict) else {}
    ar_in = ar_in if isinstance(ar_in, dict) else {}
    anomaly_result = {k: ar_in.get(k) for k in _ANOMALY_KEYS}

    tl = detect_output.get("typhoon_link") if isinstance(detect_output, dict) else {}
    tl = tl if isinstance(tl, dict) else {}
    query = tl.get("query") if isinstance(tl.get("query"), dict) else {}
    candidates_in = tl.get("candidates") if isinstance(tl.get("candidates"), list) else []
    candidates: list[dict[str, Any]] = []
    for c in candidates_in:
        if not isinstance(c, dict):
            continue
        candidates.append({k: c.get(k) for k in _CANDIDATE_KEYS})

    return {
        "anomaly_result": anomaly_result,
        "typhoon_link": {
            "query": dict(query) if query else {},
            "candidates": candidates,
        },
    }


def payload_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def parse_llm_report_json(text: str) -> tuple[dict[str, Any] | None, str]:
    """解析 assistant 输出；成功返回 (obj, '')。"""
    t = (text or "").strip()
    if not t:
        return None, "empty"
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        return None, f"json_decode:{e}"
    if not isinstance(obj, dict):
        return None, "not_object"
    if not _REPORT_KEYS.issubset(obj.keys()):
        return None, f"missing_keys:{sorted(_REPORT_KEYS - set(obj.keys()))}"
    acts = obj.get("actions")
    if not isinstance(acts, list) or len(acts) < 3:
        return None, "bad_actions_need_at_least_3"
    return obj, ""


def call_dashscope_report(
    payload: dict[str, Any],
    *,
    api_key: str,
    model: str,
    max_tokens: int = 2048,
    enable_thinking: bool = False,
) -> tuple[str | None, str]:
    """
    调用百炼部署模型。返回 (assistant 文本, 错误信息)；成功时错误信息为空串。
    """
    try:
        from dashscope import Generation
    except ImportError as e:
        return None, f"请安装 dashscope（requirements.txt 已声明）: {e}"

    messages = [
        {"role": "system", "content": SYSTEM_CHATML_ANOMALY_REPORT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "api_key": api_key,
        "result_format": "message",
        "enable_thinking": enable_thinking,
        "max_tokens": int(max_tokens),
    }
    ws = os.environ.get("DASHSCOPE_WORKSPACE", "").strip()
    if ws:
        kwargs["workspace"] = ws
    resp = Generation.call(**kwargs)
    code = getattr(resp, "status_code", None)
    if code not in (200, "200", None):
        msg = getattr(resp, "message", str(resp))
        return None, f"dashscope_status={code}: {msg}"

    out = getattr(resp, "output", None)
    text = ""
    if isinstance(out, dict):
        choices = out.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            text = str(msg.get("content") or choices[0].get("text") or "")
    if not text and out is not None:
        ch = getattr(out, "choices", None)
        if ch:
            first = ch[0]
            if isinstance(first, dict):
                text = str((first.get("message") or {}).get("content") or "")
            else:
                text = str(getattr(first, "content", "") or "")
    return text.strip(), ""


def try_llm_report(
    detect_output: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
) -> tuple[dict[str, Any] | None, str, str]:
    """
    尝试生成 LLM 报告。

    返回 (parsed_dict_or_None, raw_text_or_error, fingerprint).
    """
    payload = build_user_payload_from_detect(detect_output)
    fp = payload_fingerprint(payload)

    key = (api_key or os.environ.get("DASHSCOPE_API_KEY", "") or "").strip()
    m = (model or os.environ.get("DASHSCOPE_MODEL", "") or "").strip()
    if not key:
        return None, "未配置 DASHSCOPE_API_KEY", fp
    if not m:
        return None, "未配置模型部署代号（环境变量 DASHSCOPE_MODEL）", fp

    text, err = call_dashscope_report(payload, api_key=key, model=m, max_tokens=max_tokens)
    if err:
        return None, err, fp
    if not text:
        return None, "模型返回空内容", fp

    parsed, perr = parse_llm_report_json(text)
    if parsed is None:
        return None, f"{perr}\n---\n{text[:2000]}", fp
    return parsed, text, fp
