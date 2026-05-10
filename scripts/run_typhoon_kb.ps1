$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# 1) 下载公开数据（IBTrACS）
python scripts/download_typhoon_data.py `
  --source ibtracs `
  --output data/raw/typhoon/ibtracs/ibtracs.ALL.list.v04r01.csv

# 2) 构建事件索引与检索键
python scripts/build_typhoon_kb.py `
  --source-csv data/raw/typhoon/ibtracs/ibtracs.ALL.list.v04r01.csv `
  --source-name IBTrACS `
  --source-version v04r01

# 3) 生成两个联动案例
python scripts/demo_typhoon_kb_cases.py

Write-Host "Typhoon KB complete."
