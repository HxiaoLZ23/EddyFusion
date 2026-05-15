# FastAPI `web_api`（水文热力图 API）

包名使用 **`web_api`**，避免与 `app/services`（Streamlit 内 `from services import …`）冲突。

## 启动

在**仓库根目录**、已激活的 venv 中：

```bash
pip install -r requirements.txt
python -m uvicorn web_api.main:app --reload --host 0.0.0.0 --port 8000
```

- OpenAPI / Swagger：`http://127.0.0.1:8000/docs`
- 健康检查：`GET /api/health`

## 环境变量

| 变量 | 说明 |
|------|------|
| `REALTIME_NC_POLL_DIR` | `GET /api/realtime/latest` 轮询目录；未设置时默认 `app/data/nc_uploads` |
| `GIT_SHA` | 可选，写入 `/api/health` |

## 主要路由

- `POST /api/offline/nc` — multipart 字段 `files`
- `GET /api/realtime/latest`
- `POST /api/hydro/heatmap` — 返回 `lons`/`lats`/`values`（热力图）+ `curve_data` / `feature_names`（与 `HydroInferenceService.run` 同源）；请求体见 `docs/策略_第二套前端与水文预测热力图.md` §3
- `POST /api/hydro/meta` — 缓冲 `T_hat` / `T_need`

## CORS

开发环境允许 `http://127.0.0.1:5173` 与 `http://localhost:5173`；生产请按域名收紧。
