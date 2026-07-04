# 从仓库根启动 FastAPI（勿用其它模块名，否则无 /api/eddy 等路由）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
# 排除运行时写入目录，避免 .nc 落盘触发整仓 reload 与上传请求竞态（Win 上曾见 FileNotFoundError）
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$tyEvents = Join-Path $Root "data\processed\anomaly\typhoon_kb\events.json"
if (-not (Test-Path $tyEvents)) {
  Write-Host "Typhoon KB index missing; seeding demo events..."
  & $py (Join-Path $Root "scripts\seed_typhoon_kb_demo.py")
}
& $py -m uvicorn web_api.main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude "*/app/data/nc_uploads/*" --reload-exclude "*/app/data/eddy_preview/*" --reload-exclude "*/app/data/eddy_preview/jobs/*" --reload-exclude "*/app/data/hydro_nc_cache/*"
