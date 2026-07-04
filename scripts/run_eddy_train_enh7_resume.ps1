# 7ch 断点续训（杀毒软件中断后）
Set-Location $PSScriptRoot\..
$log = "outputs/eddy_enh7/train_resume.log"
New-Item -ItemType Directory -Force -Path "outputs/eddy_enh7" | Out-Null
Write-Host "日志: $log"
python -u -m src.eddy.train --config config/eddy_enh7.yaml --resume 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -eq 0) {
    python -m src.eddy.eval --config config/eddy_enh7.yaml --ckpt outputs/eddy_enh7/best.pt --splits val,test
}
