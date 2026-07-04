# V6 Phase B0 eval：各 config val split + 写主表
# 前置：run_eddy_v6_b0_train.ps1

param(
    [ValidateSet("p0", "p1", "p2", "p1p2", "all")]
    [string]$Priority = "p0"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

function Eval-Config {
    param([string]$Cfg)
    Write-Host ">> python -m src.eddy.eval --config $Cfg --splits val"
    python -m src.eddy.eval --config $Cfg --splits val
    if ($LASTEXITCODE -ne 0) { throw "eval failed: $Cfg" }
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
    Eval-Config $cfg
}

python scripts/eddy_write_v6_b0_fair_vs_proposed.py
Write-Host "V6 Phase B0 eval ($Priority) done."
