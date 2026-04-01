from __future__ import annotations

from typing import Any

from src.hydro.convlstm import HydroBaseline


def build_model(cfg: dict[str, Any]) -> HydroBaseline:
    d = cfg["data"]
    m = cfg["model"]
    in_c = len(d["input_features"])
    out_c = len(d["target_features"])
    t_in = int(d["input_steps"])
    t_out = int(d["output_steps"])
    return HydroBaseline(
        in_channels=in_c,
        out_channels=out_c,
        t_in=t_in,
        t_out=t_out,
        hidden_dim=int(m["hidden_dim"]),
        num_layers=int(m.get("num_layers", 2)),
        kernel_size=int(m["kernel_size"]),
        dropout=float(m["dropout"]),
        use_element_attention=bool(m.get("use_element_attention", True)),
        element_attention_hidden=int(m.get("element_attention_hidden", 64)),
    )
