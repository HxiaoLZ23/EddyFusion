from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# 保证在 streamlit 以 app 目录启动时也能找到 src/*
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hydro.dataset import HydroNpzDataset
from src.hydro.model import build_model
from src.preprocess.hydro_nc_infer_build import (
    build_hydro_xy_from_netcdf_paths,
    peek_total_time_steps,
    required_window_length,
)
from src.utils.config import load_yaml, pick_device, resolve_path


def _project_root() -> Path:
    return REPO_ROOT


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


@dataclass(frozen=True)
class HydroPreset:
    key: str
    label: str
    config_path: str
    ckpt_path: str
    metrics_paths: tuple[str, ...]


PRESETS: dict[str, HydroPreset] = {
    "l2": HydroPreset(
        key="l2",
        label="Hydro L2（默认稳定）",
        config_path="config/hydro_hycom_l2.yaml",
        ckpt_path="outputs/hydro_l2/best.pt",
        metrics_paths=(
            "outputs/hydro_l2/metrics_summary_val.json",
            "outputs/hydro_l2/metrics_summary_test.json",
        ),
    ),
    "l1": HydroPreset(
        key="l1",
        label="Hydro L1（轻量创新）",
        config_path="config/hydro_hycom_l1.yaml",
        ckpt_path="outputs/hydro_l1/best.pt",
        metrics_paths=(
            "outputs/hydro_l1/metrics_summary_val.json",
            "outputs/hydro_l1/metrics_summary_test.json",
        ),
    ),
    "l0": HydroPreset(
        key="l0",
        label="Hydro L0（实验开关）",
        config_path="config/hydro_hycom_l0.yaml",
        ckpt_path="outputs/hydro_l0/best.pt",
        metrics_paths=(
            "outputs/hydro_l0/metrics_summary_val.json",
            "outputs/hydro_l0/metrics_summary_test.json",
        ),
    ),
}


class HydroInferenceService:
    def _normalize_pred_layout(
        self,
        *,
        pred: torch.Tensor,
        gt: torch.Tensor,
        expected_out: int,
    ) -> tuple[torch.Tensor, list[str]]:
        notes: list[str] = []
        if pred.ndim != 4:
            raise ValueError(f"模型输出维度异常: pred.shape={tuple(pred.shape)}，期望 4 维 (T,C,H,W)")
        if tuple(pred.shape) == tuple(gt.shape):
            return pred, notes
        # 常见布局差异：C,T,H,W -> T,C,H,W
        if pred.shape[0] == expected_out and pred.shape[1] == gt.shape[0]:
            pred = pred.permute(1, 0, 2, 3).contiguous()
            notes.append("模型输出从 (C,T,H,W) 转为 (T,C,H,W)")
            return pred, notes
        # 若时间维与 gt 一致但通道多/少，后续按通道截断或补零
        if pred.shape[0] == gt.shape[0]:
            return pred, notes
        # 配置 output_steps 与标签 NPZ 时间长度不一致时（如 72 步预测 vs 4 步标注）：沿时间均匀重采样
        if (
            pred.ndim == 4
            and gt.ndim == 4
            and pred.shape[1:] == gt.shape[1:]
            and pred.shape[0] != gt.shape[0]
            and gt.shape[0] >= 1
        ):
            t_out, t_gt = int(pred.shape[0]), int(gt.shape[0])
            idx = torch.linspace(0, t_out - 1, t_gt).long().clamp(0, t_out - 1)
            pred = pred[idx].contiguous()
            notes.append(f"时间维从模型输出 T={t_out} 重采样对齐到标签 T={t_gt}")
            return pred, notes
        raise ValueError(
            f"模型输出形状与标签不兼容: pred={tuple(pred.shape)}, gt={tuple(gt.shape)}，无法自动对齐。"
        )

    def feature_requirements(self, *, config_path: str) -> dict[str, Any]:
        cfg = load_yaml(config_path)
        in_feats = list(cfg.get("data", {}).get("input_features", []))
        out_feats = list(cfg.get("data", {}).get("target_features", []))
        return {
            "input_features": in_feats,
            "target_features": out_feats,
            "expected_in": len(in_feats),
            "expected_out": len(out_feats),
        }

    def inspect_npz_channels(self, *, npz_path: str | Path, preferred_key: str) -> dict[str, Any]:
        p = resolve_path(npz_path)
        if not p.is_file():
            return {"exists": False, "path": str(p), "channels": 0, "shape": []}
        d = np.load(p)
        key = preferred_key if preferred_key in d else d.files[0]
        arr = d[key]
        shape = list(arr.shape)
        ch = int(shape[-1]) if len(shape) >= 1 else 0
        return {"exists": True, "path": str(p), "key": key, "channels": ch, "shape": shape}

    def _apply_channel_mapping(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        x_channel_indices: list[int] | None,
        y_channel_indices: list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        notes: list[str] = []
        if x_channel_indices:
            x_max = int(x.shape[1])
            if any((i < 0 or i >= x_max) for i in x_channel_indices):
                raise ValueError(f"X 通道映射越界：可用范围 [0,{x_max - 1}]，收到 {x_channel_indices}")
            idx = torch.tensor(x_channel_indices, dtype=torch.long)
            x = x.index_select(1, idx)
            notes.append(f"X 按映射通道重排: {x_channel_indices}")
        if y_channel_indices:
            y_max = int(y.shape[1])
            if any((i < 0 or i >= y_max) for i in y_channel_indices):
                raise ValueError(f"y 通道映射越界：可用范围 [0,{y_max - 1}]，收到 {y_channel_indices}")
            idx = torch.tensor(y_channel_indices, dtype=torch.long)
            y = y.index_select(1, idx)
            notes.append(f"y 按映射通道重排: {y_channel_indices}")
        return x, y, notes

    def _align_channels(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        expected_in: int,
        expected_out: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        notes: list[str] = []
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError(f"样本维度异常：x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}，期望均为 (T,C,H,W)")
        x_ch = int(x.shape[1])
        y_ch = int(y.shape[1])
        if x_ch != expected_in:
            if x_ch > expected_in:
                x = x[:, :expected_in, :, :]
                notes.append(f"X 通道从 {x_ch} 截断到 {expected_in}")
            else:
                pad = torch.zeros((x.shape[0], expected_in - x_ch, x.shape[2], x.shape[3]), dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
                notes.append(f"X 通道从 {x_ch} 补零到 {expected_in}")
        if y_ch != expected_out:
            if y_ch > expected_out:
                y = y[:, :expected_out, :, :]
                notes.append(f"y 通道从 {y_ch} 截断到 {expected_out}")
            else:
                pad = torch.zeros((y.shape[0], expected_out - y_ch, y.shape[2], y.shape[3]), dtype=y.dtype)
                y = torch.cat([y, pad], dim=1)
                notes.append(f"y 通道从 {y_ch} 补零到 {expected_out}")
        return x, y, notes

    def available_presets(self, *, enable_l0: bool) -> list[HydroPreset]:
        keys = ["l2", "l1"] + (["l0"] if enable_l0 else [])
        return [PRESETS[k] for k in keys]

    def validate_paths(self, *, config_path: str, ckpt_path: str) -> dict[str, Any]:
        cp = resolve_path(config_path)
        kp = resolve_path(ckpt_path)
        out: dict[str, Any] = {
            "config_exists": cp.is_file(),
            "ckpt_exists": kp.is_file(),
            "config_path": str(cp),
            "ckpt_path": str(kp),
            "messages": [],
        }
        if not cp.is_file():
            out["messages"].append(f"配置文件不存在: {cp}")
        if not kp.is_file():
            out["messages"].append(f"权重文件不存在: {kp}")
        return out

    def resolve_dataset_paths(
        self,
        *,
        cfg: dict[str, Any],
        split: str,
        x_path_override: str | None = None,
        y_path_override: str | None = None,
    ) -> tuple[Path, Path]:
        if x_path_override and y_path_override:
            return resolve_path(x_path_override), resolve_path(y_path_override)
        sx = f"{split}_data"
        sy = f"{split}_label"
        return resolve_path(cfg["paths"][sx]), resolve_path(cfg["paths"][sy])

    def run(
        self,
        *,
        config_path: str,
        ckpt_path: str,
        split: str,
        sample_index: int,
        x_path_override: str | None = None,
        y_path_override: str | None = None,
        x_channel_indices: list[int] | None = None,
        y_channel_indices: list[int] | None = None,
        map_time_index: int | None = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        cfg = load_yaml(config_path)
        level = int(cfg.get("meta", {}).get("level", -1))

        x_path, y_path = self.resolve_dataset_paths(
            cfg=cfg,
            split=split,
            x_path_override=x_path_override,
            y_path_override=y_path_override,
        )
        if not x_path.is_file() or not y_path.is_file():
            missing = []
            if not x_path.is_file():
                missing.append(str(x_path))
            if not y_path.is_file():
                missing.append(str(y_path))
            raise FileNotFoundError("输入数据不存在: " + " | ".join(missing))

        ds = HydroNpzDataset(x_path, y_path)
        if len(ds) == 0:
            raise ValueError("数据集样本数为 0，无法推理。")
        idx = max(0, min(int(sample_index), len(ds) - 1))
        x, y = ds[idx]
        x, y, map_notes = self._apply_channel_mapping(
            x=x,
            y=y,
            x_channel_indices=x_channel_indices,
            y_channel_indices=y_channel_indices,
        )
        feature_in = len(cfg["data"]["input_features"])
        feature_out = len(cfg["data"]["target_features"])
        x, y, channel_notes = self._align_channels(x=x, y=y, expected_in=feature_in, expected_out=feature_out)

        device = torch.device(pick_device(str(cfg.get("train", {}).get("device", "cuda"))))
        model = build_model(cfg).to(device)
        ckpt = resolve_path(ckpt_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"未找到权重: {ckpt}")
        try:
            state = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()

        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).detach().float().cpu()[0]
        gt = y.detach().float().cpu()
        pred, pred_layout_notes = self._normalize_pred_layout(pred=pred, gt=gt, expected_out=feature_out)
        if pred.shape[1] != feature_out:
            pred_ch = int(pred.shape[1])
            if pred.shape[1] > feature_out:
                pred = pred[:, :feature_out, :, :]
                pred_layout_notes.append(f"模型输出通道从 {pred_ch} 截断到 {feature_out}")
            else:
                pad = torch.zeros((pred.shape[0], feature_out - pred.shape[1], pred.shape[2], pred.shape[3]), dtype=pred.dtype)
                pred = torch.cat([pred, pad], dim=1)
                pred_layout_notes.append(f"模型输出通道从 {pred_ch} 补零到 {feature_out}")
        # T C H W
        diff = pred - gt
        se = (diff**2).mean(dim=(0, 2, 3)).numpy()
        ae = gt.abs().mean(dim=(0, 2, 3)).numpy()
        rmse = np.sqrt(se)
        nrmse = rmse / np.maximum(ae, 1e-6)
        names = list(cfg["data"]["target_features"])
        rmse_map = {names[i]: _safe_float(rmse[i]) for i in range(len(names))}
        nrmse_map = {names[i]: _safe_float(nrmse[i]) for i in range(len(names))}

        t_max = int(gt.shape[0] - 1)
        if map_time_index is None:
            t_map = t_max
        else:
            t_map = max(0, min(int(map_time_index), t_max))
        curve_data: dict[str, list[dict[str, float]]] = {}
        map_data: dict[str, dict[str, np.ndarray]] = {}
        for i, name in enumerate(names):
            gt_curve = gt[:, i, :, :].mean(dim=(1, 2)).numpy()
            pd_curve = pred[:, i, :, :].mean(dim=(1, 2)).numpy()
            feature_rows = []
            for t_idx in range(len(gt_curve)):
                feature_rows.append(
                    {
                        "horizon": float(t_idx + 1),
                        "gt": _safe_float(gt_curve[t_idx]),
                        "pred": _safe_float(pd_curve[t_idx]),
                    }
                )
            curve_data[name] = feature_rows
            gt_map = gt[t_map, i, :, :].numpy()
            pd_map = pred[t_map, i, :, :].numpy()
            err_map = np.abs(pd_map - gt_map)
            map_data[name] = {"gt": gt_map, "pred": pd_map, "err": err_map}

        return {
            "status": "success",
            "split": split,
            "sample_index": idx,
            "dataset_size": len(ds),
            "level": level,
            "device": str(device),
            "nrmse_avg": _safe_float(np.mean(nrmse)),
            "rmse_per_feature": rmse_map,
            "nrmse_per_feature": nrmse_map,
            "curve_data": curve_data,
            "map_data": map_data,
            "feature_names": names,
            "t_last": t_map + 1,
            "elapsed_sec": _safe_float(time.time() - t0),
            "warnings": map_notes + channel_notes + pred_layout_notes,
        }

    def load_metrics_summary(self, metrics_paths: tuple[str, ...]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in metrics_paths:
            fp = resolve_path(p)
            if not fp.is_file():
                out.append({"path": str(fp), "exists": False, "raw": {}, "message": "missing"})
                continue
            try:
                import json

                raw = json.loads(fp.read_text(encoding="utf-8"))
                out.append({"path": str(fp), "exists": True, "raw": raw, "message": "ok"})
            except Exception as e:
                out.append({"path": str(fp), "exists": False, "raw": {}, "message": f"read_error: {e}"})
        return out

    def save_uploaded_npz(self, upload_file: Any) -> Path:
        suffix = Path(upload_file.name).suffix.lower()
        if suffix != ".npz":
            raise ValueError("仅支持 .npz 文件")
        root = _project_root() / "app" / "data" / "hydro_inputs"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz", dir=root) as tmp:
            tmp.write(upload_file.getvalue())
            return Path(tmp.name)

    def hydro_required_time_steps(self, config_path: str) -> int:
        cfg = load_yaml(config_path)
        return required_window_length(cfg)

    def peek_hydro_buffer_time_steps(self, nc_paths: list[Path], *, config_path: str) -> int:
        cfg = load_yaml(config_path)
        feats = list(cfg["data"]["input_features"])
        return peek_total_time_steps(nc_paths, feats)

    def materialize_netcdf_to_xy_npz(
        self,
        nc_paths: list[Path],
        *,
        config_path: str,
        window_stride: int = 1,
        max_windows: int | None = 256,
    ) -> tuple[Path, Path, dict[str, Any]]:
        """
        将多个 NetCDF 拼接并滑窗，写入 `app/data/hydro_nc_cache/` 下临时 X/y.npz。
        """
        cfg = load_yaml(config_path)
        x, y, meta = build_hydro_xy_from_netcdf_paths(
            nc_paths,
            cfg,
            window_stride=int(window_stride),
            max_windows=max_windows,
        )
        cache = _project_root() / "app" / "data" / "hydro_nc_cache"
        cache.mkdir(parents=True, exist_ok=True)
        tag = f"{int(time.time() * 1000)}"
        xp = cache / f"X_nc_{tag}.npz"
        yp = cache / f"y_nc_{tag}.npz"
        np.savez_compressed(xp, X=x)
        np.savez_compressed(yp, y=y)
        meta["x_path"] = str(xp)
        meta["y_path"] = str(yp)
        return xp, yp, meta
