#!/usr/bin/env python3
"""生成 config/eddy_ablation/*.yaml（多通道消融，init_from_baseline）。"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 3ch 增通道消融：单路叠加 + 双路组合 + 7ch 全量（见 docs/实验与结果归档/涡旋_7ch通道消融计划.md）
PROFILES = (
    "4_bgr_zeta",
    "4_bgr_ow",
    "5_bgr_grad",
    "5_no_grad",
    "6_no_ow",
    "6_no_zeta",
)

TEMPLATE = """# 7ch 通道消融：{profile}（channels={ch}）
meta:
  project: "EddyFusion"
  module: "eddy"
  version: "1.0-abl-{profile}"
  level: 2
  seed: 42

paths:
  dataset_yaml: "AutoDL/dataset/eddy_ablation/{profile}/dataset.yaml"
  output_dir: "outputs/eddy_ablation/{profile}"

model:
  backbone: "yolov8n-seg"
  architecture_yaml: "yolov8n-seg.yaml"
  input_size: [640, 640]
  channels: {ch}
  init_from_baseline: "outputs/eddy_cloud_fair/last.pt"

train:
  device: "cuda"
  epochs: {epochs}
  batch_size: 4
  workers: 4
  amp: false
  pretrained: false
  run_name: train
  seed: 42
  lr0: 0.0025
  lrf: 0.01
  warmup_epochs: 12
  cos_lr: true
  close_mosaic: 20
  patience: 150
  mosaic: 0.25
  mixup: 0.0
  copy_paste: 0.0
  label_smoothing: 0.05
  degrees: 0.0
  translate: 0.05
  scale: 0.4
  fliplr: 0.5

eval:
  metrics_file: "outputs/eddy_ablation/{profile}/metrics_summary.json"
"""


def main() -> None:
    import argparse

    from src.eddy.stacked_physics import ablation_profile_channels

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    out_dir = REPO / "config" / "eddy_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for profile in PROFILES:
        ch = ablation_profile_channels(profile)
        text = TEMPLATE.format(profile=profile, ch=ch, epochs=int(args.epochs))
        path = out_dir / f"{profile}.yaml"
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)} (channels={ch})")


if __name__ == "__main__":
    main()
