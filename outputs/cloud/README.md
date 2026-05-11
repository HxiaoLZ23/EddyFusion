# `outputs/cloud/` — 水文专项 A 云端评估产物（归档约定）

与云机同步目录 **`AutoDL/outputs/cloud/`** 存放内容一致；可将审计 JSON、compare summary 拷至此目录便于本地查阅。**大 JSON 可能被 `.gitignore` 忽略，本 README 可提交以固定命名约定。**

建议文件名：

| 文件 | 来源命令 |
|------|----------|
| `hydro_preprocess_audit.json` | `python scripts/hydro_cloud_assessment.py audit ... --out-json outputs/cloud/hydro_preprocess_audit.json` |
| `hydro_compare_val_summary.json` | `compare --split val ... --out-summary-json ...` |
| `hydro_compare_test_summary.json` | `compare --split test ...`（与 val **分机**） |

专项 B 多轮实验可加后缀，例如 `hydro_compare_val_summary_eos005.json`。

详见 **`docs/水文_云端归档与专项B启动.md`**。
