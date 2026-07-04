# V6 Phase A 数据导出（2018Q1 train + 2024 val，OW P24，方案 A norm）
# 规范：开发阶段文档/基于时空一致性弱监督标签的中尺度涡旋实例分割训练方案v6.md

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

$Common = @(
    "--time-stride", "1",
    "--single-percentile", "24",
    "--skip-boundary-days",
    "--no-test-split",
    "--rgb-p-lo", "2",
    "--rgb-p-hi", "98"
)

$TrainNC = "20130101_20221231"
$TrainStart = "2018-01-01"
$TrainEnd = "2018-03-31"

function Export-Split {
    param(
        [string]$Out,
        [string]$Mode,
        [string]$Split,
        [string]$NcStem = "",
        [string]$TimeStart = "",
        [string]$TimeEnd = "",
        [switch]$SkipLabels,
        [string]$CopyLabelsFrom = ""
    )
    $args = @(
        "-m", "src.preprocess.eddy_yolo_export",
        "--out", $Out,
        "--input-mode", $Mode,
        "--split", $Split
    ) + $Common
    if ($NcStem) { $args += @("--nc-stem", $NcStem) }
    if ($TimeStart) { $args += @("--time-start", $TimeStart) }
    if ($TimeEnd) { $args += @("--time-end", $TimeEnd) }
    if ($SkipLabels) { $args += "--skip-labels" }
    if ($CopyLabelsFrom) { $args += @("--copy-labels-from", $CopyLabelsFrom) }
    Write-Host ">> python $($args -join ' ')"
    python @args
    if ($LASTEXITCODE -ne 0) { throw "export failed: $Out $Mode $Split" }
}

$Leakage = "AutoDL/dataset/eddy_v6_leakage"
$Fair = "AutoDL/dataset/eddy_v6_fair"
$Proposed = "AutoDL/dataset/eddy_v6_proposed"

# 1) Leakage：train 2018Q1 + val 2024（含 labels）
Export-Split -Out $Leakage -Mode leakage -Split train -NcStem $TrainNC -TimeStart $TrainStart -TimeEnd $TrainEnd
Export-Split -Out $Leakage -Mode leakage -Split val

# 2) Fair / Proposed：images only，labels 从 leakage 复制
foreach ($pair in @(
        @($Fair, "fair"),
        @($Proposed, "triplet")
    )) {
    $out, $mode = $pair
    Export-Split -Out $out -Mode $mode -Split train -NcStem $TrainNC -TimeStart $TrainStart -TimeEnd $TrainEnd -SkipLabels -CopyLabelsFrom $Leakage
    Export-Split -Out $out -Mode $mode -Split val -SkipLabels -CopyLabelsFrom $Leakage
}

Write-Host "V6 Phase A export done."
