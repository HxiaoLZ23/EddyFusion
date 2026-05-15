# EddyFusion 第二套前端（Vite + React）

与 Streamlit 并行，专注水文预测热力图 + MapLibre。

## 环境

- Node 20+（建议 LTS）
- 后端：`web_api` FastAPI（见 `../web_api/README.md`）

## 配置

复制 `.env.development`，设置：

- `VITE_API_BASE`：FastAPI 根地址，如 `http://127.0.0.1:8000`

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
