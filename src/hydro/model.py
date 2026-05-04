from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.hydro.attn_residual import build_attn_neck
from src.hydro.convlstm import HydroBaseline
from src.hydro.mot_stem import build_mot_stem


class HydroMotCascade(nn.Module):
    """可选 MoT 时序 Stem + ConvLSTM 主干。"""

    def __init__(self, stem: nn.Module | None, backbone: HydroBaseline) -> None:
        super().__init__()
        self.stem = stem if stem is not None else nn.Identity()
        self.backbone = backbone

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.backbone(self.stem(x))


def build_model(cfg: dict[str, Any]) -> nn.Module:
    d = cfg["data"]
    m = cfg["model"]
    in_c = len(d["input_features"])
    out_c = len(d["target_features"])
    t_in = int(d["input_steps"])
    t_out = int(d["output_steps"])
    hd = int(m["hidden_dim"])
    backbone = HydroBaseline(
        in_channels=in_c,
        out_channels=out_c,
        t_in=t_in,
        t_out=t_out,
        hidden_dim=hd,
        num_layers=int(m.get("num_layers", 2)),
        kernel_size=int(m["kernel_size"]),
        dropout=float(m["dropout"]),
        use_element_attention=bool(m.get("use_element_attention", True)),
        element_attention_hidden=int(m.get("element_attention_hidden", 64)),
        use_encoder_checkpoint=bool(m.get("encoder_checkpoint", False)),
        attn_neck=build_attn_neck(cfg, hidden_dim=hd),
    )
    stem = build_mot_stem(cfg, in_channels=in_c)
    if stem is None:
        return backbone
    return HydroMotCascade(stem, backbone)
