# EddyFusion 第二套前端（Vite + React）

与 Streamlit 并行，专注水文预测热力图 + MapLibre。

## 环境

- Node 20+（建议 LTS）
- 后端：`web_api` FastAPI（见 `../web_api/README.md`）

## 配置

复制 `.env.development`，设置：

- `VITE_API_BASE`：FastAPI 根地址，如 `http://127.0.0.1:8000`
- `VITE_SHOW_HYDRO`：设为 `true` 时展示水文大屏区块与 `/l1/hydro`；**默认不设置即关闭**（待新水文模型后再开）。见 `docs/实验与结果归档/水文_其他指标与能用标准归档.md` §5。

## 脚本

```bash
npm install
npm run dev
```

浏览器访问 `/offline`、`/realtime`。

```bash
npm run build
```

产物在 `dist/`，可由 Nginx 或 FastAPI `StaticFiles` 挂载。
