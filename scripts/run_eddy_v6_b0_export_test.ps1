# V6 Fair-B0：补导出 test 2023（20230101_20231231，k_max=5 与 train/val 同口径）
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

$Common = @(
    "--time-stride", "1",
    "--single-percentile", "24",
    "--center-offset-max", "5",
    "--rgb-p-lo", "2",
    "--rgb-p-hi", "98"
)

$Leakage = "AutoDL/dataset/eddy_v6_b0_leakage"
$Fair = "AutoDL/dataset/eddy_v6_b0_fair"

Write-Host ">> leakage test (labels + images)"
python -m src.preprocess.eddy_yolo_export `
    --out $Leakage `
    --input-mode leakage `
    --split test `
    --triplet-offset 1 `
    @Common
if ($LASTEXITCODE -ne 0) { throw "leakage test export failed" }

Write-Host ">> fair test (images only, labels from leakage)"
python -m src.preprocess.eddy_yolo_export `
    --out $Fair `
    --input-mode fair `
    --split test `
    --triplet-offset 1 `
    --skip-labels `
    --copy-labels-from $Leakage `
    @Common
if ($LASTEXITCODE -ne 0) { throw "fair test export failed" }

Write-Host ">> eval Fair-B0 on test"
python -m src.eddy.eval --config config/eddy_v6_b0_fair.yaml --ckpt outputs/eddy_v6_b0_fair/best.pt --splits test

Write-Host "Done. See outputs/eddy_v6_b0_fair/metrics_summary_test.json"
