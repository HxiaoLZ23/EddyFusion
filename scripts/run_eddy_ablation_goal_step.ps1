# /goal 单步：跑动态规划的下一个 profile（Full），完成后 record + 更新表
# Usage:
#   .\scripts\run_eddy_ablation_goal_step.ps1
#   .\scripts\run_eddy_ablation_goal_step.ps1 -RecordOnly -Profile 4_bgr_zeta
param(
    [switch]$RecordOnly,
    [string]$Profile = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($RecordOnly) {
    if ($Profile -eq "") {
        Write-Error "RecordOnly requires -Profile"
    }
    python scripts/eddy_ablation_dynamic.py record --profile $Profile
    python scripts/eddy_write_ablation_map_table.py
    python scripts/eddy_ablation_dynamic.py status
    exit $LASTEXITCODE
}

$next = python scripts/eddy_ablation_dynamic.py next 2>&1 | Out-String
$next = $next.Trim()
if ($next -eq "DONE") {
    Write-Host "Dynamic plan DONE. See submission/tables/eddy_ablation_dynamic_plan.md"
    python scripts/eddy_write_ablation_map_table.py
    exit 0
}

Write-Host "== goal step: profile=$next =="
.\scripts\run_eddy_7ch_ablation_local.ps1 -Full -Profile $next
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/eddy_ablation_dynamic.py record --profile $next
python scripts/eddy_write_ablation_map_table.py
python scripts/eddy_ablation_dynamic.py status
