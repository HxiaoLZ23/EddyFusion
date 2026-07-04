# V6 Phase B0 导出：2018 全年 train + 2024 val，k_max=5 交集 stem（355/356）
# 规范：开发阶段文档/基于时空一致性弱监督标签的中尺度涡旋实例分割训练方案v6.md

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

$CenterOffsetMax = 5
$TrainNC = "20130101_20221231"
$TrainStart = "2018-01-01"
$TrainEnd = "2018-12-31"

$Common = @(
    "--time-stride", "1",
    "--single-percentile", "24",
    "--center-offset-max", "$CenterOffsetMax",
    "--no-test-split",
    "--rgb-p-lo", "2",
    "--rgb-p-hi", "98"
)

function Export-Split {
    param(
        [string]$Out,
        [string]$Mode,
        [string]$Split,
        [int]$TripletOffset = 1,
        [string]$NcStem = "",
        [string]$TimeStart = "",
        [string]$TimeEnd = "",
        [switch]$SkipLabels,
        [string]$CopyLabelsFrom = ""
    )
    $pyArgs = @(
        "-m", "src.preprocess.eddy_yolo_export",
        "--out", $Out,
        "--input-mode", $Mode,
        "--split", $Split,
        "--triplet-offset", "$TripletOffset"
    ) + $Common
    if ($NcStem) { $pyArgs += @("--nc-stem", $NcStem) }
    if ($TimeStart) { $pyArgs += @("--time-start", $TimeStart) }
    if ($TimeEnd) { $pyArgs += @("--time-end", $TimeEnd) }
    if ($SkipLabels) { $pyArgs += "--skip-labels" }
    if ($CopyLabelsFrom) { $pyArgs += @("--copy-labels-from", $CopyLabelsFrom) }
    Write-Host ">> python $($pyArgs -join ' ')"
    python @pyArgs
    if ($LASTEXITCODE -ne 0) { throw "export failed: $Out $Mode $Split" }
}

$Leakage = "AutoDL/dataset/eddy_v6_b0_leakage"
$Fair = "AutoDL/dataset/eddy_v6_b0_fair"
$K1 = "AutoDL/dataset/eddy_v6_b0_proposed_k1"
$K3 = "AutoDL/dataset/eddy_v6_b0_proposed_k3"
$K5 = "AutoDL/dataset/eddy_v6_b0_proposed_k5"

# 1) Leakage：labels + images（train 2018 + val 2024；val 不用 skip-boundary-days）
Export-Split -Out $Leakage -Mode leakage -Split train -NcStem $TrainNC -TimeStart $TrainStart -TimeEnd $TrainEnd
Export-Split -Out $Leakage -Mode leakage -Split val

# 2) 其余 mode：同 stem labels，仅 images 不同
foreach ($item in @(
        @($Fair, "fair", 1),
        @($K1, "triplet", 1),
        @($K3, "triplet", 3),
        @($K5, "triplet", 5)
    )) {
    $out, $mode, $k = $item
    Export-Split -Out $out -Mode $mode -TripletOffset $k -Split train `
        -NcStem $TrainNC -TimeStart $TrainStart -TimeEnd $TrainEnd `
        -SkipLabels -CopyLabelsFrom $Leakage
    Export-Split -Out $out -Mode $mode -TripletOffset $k -Split val `
        -SkipLabels -CopyLabelsFrom $Leakage
}

python scripts/eddy_v6_b0_verify_stems.py `
    --datasets $Leakage,$Fair,$K1,$K3,$K5 `
    --manifest submission/tables/eddy_v6_b0_stem_manifest.json `
    --expect-train 355 --expect-val 356
if ($LASTEXITCODE -ne 0) { throw "stem verify failed" }

Write-Host "V6 Phase B0 export done."
