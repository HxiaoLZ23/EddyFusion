# 云端同口径本地公平对比：3ch -> 7ch -> eval（AutoDL/dataset，val=53）
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

New-Item -ItemType Directory -Force -Path "outputs/eddy_cloud_fair","outputs/eddy_enh7_cloud_fair" | Out-Null

Write-Host "== [1/4] 3ch train (AutoDL/dataset/eddy) ==" -ForegroundColor Cyan
python -u -m src.eddy.train --config config/eddy_cloud_fair.yaml 2>&1 | Tee-Object -FilePath "outputs/eddy_cloud_fair/train.log" -Append

Write-Host "== [2/4] 7ch train (AutoDL/dataset/eddy_enh7) ==" -ForegroundColor Cyan
python -u -m src.eddy.train --config config/eddy_enh7_cloud_fair.yaml 2>&1 | Tee-Object -FilePath "outputs/eddy_enh7_cloud_fair/train.log"

Write-Host "== [3/4] eval val/test ==" -ForegroundColor Cyan
python -m src.eddy.eval --config config/eddy_cloud_fair.yaml --ckpt outputs/eddy_cloud_fair/best.pt --splits val,test
python -m src.eddy.eval --config config/eddy_enh7_cloud_fair.yaml --ckpt outputs/eddy_enh7_cloud_fair/best.pt --splits val,test

Write-Host "== [4/4] 对比 ==" -ForegroundColor Cyan
python -c @"
import json
from pathlib import Path
def load(split, base):
    p = Path(base).parent / f"metrics_summary_{split}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["metrics"]["mask_map50"]
for tag, base in [("3ch", "outputs/eddy_cloud_fair/metrics_summary_val.json"), ("7ch", "outputs/eddy_enh7_cloud_fair/metrics_summary_val.json")]:
    b = Path(base)
    pv = load("val", b)
    pt = load("test", b)
    print(f"{tag}  val={pv:.4f}  test={pt:.4f}")
"@

Write-Host "完成" -ForegroundColor Green
