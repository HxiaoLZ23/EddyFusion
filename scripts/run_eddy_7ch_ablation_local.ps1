# 7ch channel ablation: export -> train -> eval (Windows)
# Usage:
#   .\scripts\run_eddy_7ch_ablation_local.ps1 -Smoke
#   .\scripts\run_eddy_7ch_ablation_local.ps1 -Full
#   .\scripts\run_eddy_7ch_ablation_local.ps1 -ExportOnly
#   .\scripts\run_eddy_7ch_ablation_local.ps1 -TrainOnly -Full
param(
    [switch]$Smoke,
    [switch]$Full,
    [switch]$ExportOnly,
    [switch]$TrainOnly,
    [string]$Profile = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $Smoke -and -not $Full) {
    if (-not $ExportOnly -and -not $TrainOnly) {
        $Smoke = $true
        Write-Host "Default: -Smoke"
    }
}

$Epochs = if ($Full) { 100 } else { 5 }
$Stride = 7
$MaxFrames = if ($Smoke) { 2 } else { $null }
$AllProfiles = @(
    "4_bgr_zeta", "4_bgr_ow", "5_bgr_grad",
    "5_no_grad", "6_no_ow", "6_no_zeta"
)
if ($Profile -ne "") {
    if ($AllProfiles -notcontains $Profile) {
        Write-Error "Unknown -Profile '$Profile'. Valid: $($AllProfiles -join ', ')"
    }
    $Profiles = @($Profile)
} else {
    $Profiles = $AllProfiles
}

python scripts/gen_eddy_ablation_configs.py --epochs $Epochs

if (-not $TrainOnly) {
    foreach ($p in $Profiles) {
        $out = "AutoDL/dataset/eddy_ablation/$p"
        Write-Host "== export profile=$p -> $out =="
        $exportArgs = @(
            "-m", "src.preprocess.eddy_yolo_export",
            "--data-config", "config/data.yaml",
            "--out", $out,
            "--stack-physics-npy", "--stack-profile", $p,
            "--time-stride", "$Stride"
        )
        if ($null -ne $MaxFrames) {
            $exportArgs += @("--max-frames-per-file", "$MaxFrames")
        }
        python @exportArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        python scripts/check_eddy_ready.py --dataset-yaml "$out/dataset.yaml"
    }
}

if ($ExportOnly) {
    Write-Host "ExportOnly done."
    exit 0
}

foreach ($p in $Profiles) {
    $cfg = "config/eddy_ablation/$p.yaml"
    Write-Host "== train $p n_epochs=$Epochs =="
    python -u -m src.eddy.train --config $cfg
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $ckpt = "outputs/eddy_ablation/$p/best.pt"
    if (-not (Test-Path $ckpt)) {
        $ckpt = "outputs/eddy_ablation/$p/train/weights/best.pt"
    }
    Write-Host "== eval $p =="
    python -m src.eddy.eval --config $cfg --ckpt $ckpt --splits val,test
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python scripts/eddy_write_ablation_map_table.py
Write-Host "Done: submission/tables/eddy_ablation_7ch_matrix.md"
