# 从仓库根启动 FastAPI（勿用其它模块名，否则无 /api/eddy 等路由）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
# 排除运行时写入目录，避免 .nc 落盘触发整仓 reload 与上传请求竞态（Win 上曾见 FileNotFoundError）
python -m uvicorn web_api.main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude "*/app/data/nc_uploads/*" --reload-exclude "*/app/data/eddy_preview/*" --reload-exclude "*/app/data/hydro_nc_cache/*"
