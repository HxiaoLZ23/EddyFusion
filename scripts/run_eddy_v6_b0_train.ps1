# V6 Phase B0 训练：P0 Fair + Proposed-k3；可选 P1/P2
# 前置：scripts/run_eddy_v6_phase_b0_export.ps1

param(
    [ValidateSet("p0", "p1", "p2", "p1p2", "all")]
    [string]$Priority = "p0"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

function Train-Config {
    param([string]$Cfg)
    Write-Host ">> python -m src.eddy.train --config $Cfg"
    python -m src.eddy.train --config $Cfg
    if ($LASTEXITCODE -ne 0) { throw "train failed: $Cfg" }
}

$p0 = @("config/eddy_v6_b0_fair.yaml", "config/eddy_v6_b0_proposed_k3.yaml")
$p1 = @("config/eddy_v6_b0_proposed_k1.yaml")
$p2 = @("config/eddy_v6_b0_leakage.yaml", "config/eddy_v6_b0_proposed_k5.yaml")

$queue = switch ($Priority) {
    "p0" { $p0 }
    "p1" { $p1 }
    "p2" { $p2 }
    "p1p2" { $p1 + $p2 }
    "all" { $p0 + $p1 + $p2 }
}

foreach ($cfg in $queue) {
    Train-Config $cfg
}

Write-Host "V6 Phase B0 train ($Priority) done."
