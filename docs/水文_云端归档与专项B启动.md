# 水文专项 A 归档路径 + 专项 B（L0 优化）启动说明

## 1. 云端产物归档（专项 A：`audit` / `compare`）

**两套等价布局（二选一或同时存在）：**

| 位置 | 说明 |
|------|------|
| **`AutoDL/outputs/cloud/`** | 云机结果下载到仓库旁的常用路径；**整目录 `AutoDL/` 默认 `.gitignore`**，不向远程提交 JSON。 |
| **`outputs/cloud/`** | 仓库内可读路径；其中 **`README.md`** 已纳入版本库，说明应放入的文件名（见该文件）。 |

**建议固定文件名（与 `hydro_cloud_assessment.py --out-json` / `--out-summary-json` 一致）：**

- `hydro_preprocess_audit.json` — `audit` 输出；**`issues` 为空**即视为预处理与统计量自检通过。
- `hydro_compare_val_summary.json` — `compare --split val`
- `hydro_compare_test_summary.json` — `compare --split test`（**勿与 val 共用文件名**，避免覆盖）

**对比表（材料用）：** `submission/tables/hydro_l0_vs_baseline_val.{md,csv}`、`hydro_l0_vs_baseline_test.{md,csv}`（由 `compare` 的 `--out-table-md` / CSV 选项生成，见 `docs/下一步执行清单_云端评估前端与L0优化.md` §1.5）。

副本拷入 **`AutoDL/outputs/cloud/`** 后，可在 `后续开发工作清单_未完成项与云端L0专项.md` **§2.4** 勾选「归档完成」。

---

## 2. 专项 B 启动声明

- **前提**：专项 A **实质结论**已在清单 **§2.0 / §3.0**（数据未被列为不可信；L0 vs L2 扩展指标已有）。
- **启动含义**：按 **`后续开发工作清单` §3.1** 开始做 **低成本成组对照**（每次只改 1～2 个超参），并用 **`hydro_cloud_assessment.py compare`** 与 **L2 `best.pt`** 对齐验收。
- **主指标**：答辩前在 **§3.4 口径 A（误差+Pearson）** 与 **口径 B（含 Skill）** 中书面锁定其一或组合规则。

---

## 3. 首轮实验建议（§3.1 ① EOS 权重）

**实验 ID**：`B-3.1-eos005`（示例）

**配置**：`config/experiments/hydro_hycom_l0_eos005.yaml`  
相对基线 `hydro_hycom_l0.yaml` **仅** `loss.eos_constraint.weight: 0.1 → 0.05`，**输出目录** `outputs/hydro_l0_eos005`，避免覆盖原 L0 权重。

**云机（仓库根）执行：**

```bash
# 训练（专项 B 第一轮）
python -m src.hydro.train --config config/experiments/hydro_hycom_l0_eos005.yaml

# 与 L2 对比（val / test 各一次；ckpt 路径按实际产物调整）
python scripts/hydro_cloud_assessment.py compare --split val \
  --baseline-config config/hydro_hycom_l2.yaml --baseline-ckpt outputs/hydro_l2/best.pt \
  --experiment-config config/experiments/hydro_hycom_l0_eos005.yaml \
  --experiment-ckpt outputs/hydro_l0_eos005/best.pt \
  --out-table-md submission/tables/hydro_l0_eos005_vs_l2_val.md \
  --out-summary-json AutoDL/outputs/cloud/hydro_compare_val_summary_eos005.json

python scripts/hydro_cloud_assessment.py compare --split test \
  --baseline-config config/hydro_hycom_l2.yaml --baseline-ckpt outputs/hydro_l2/best.pt \
  --experiment-config config/experiments/hydro_hycom_l0_eos005.yaml \
  --experiment-ckpt outputs/hydro_l0_eos005/best.pt \
  --out-table-md submission/tables/hydro_l0_eos005_vs_l2_test.md \
  --out-summary-json AutoDL/outputs/cloud/hydro_compare_test_summary_eos005.json
```

将生成的 summary / md **拷入 `AutoDL/outputs/cloud/`**（或 `submission/tables/`）并在 **`开发阶段文档/训练与实验记录.md`**（若使用）记一行：实验 ID、config、目的、val/test 结论。

---

### 3.1 第二组：`B-3.1-eos003`（EOS weight 0.03，与 L2 对比）

**配置**：`config/experiments/hydro_hycom_l0_eos003.yaml`（`loss.eos_constraint.weight: 0.03`，`output_dir: outputs/hydro_l0_eos003`）。

**云机（仓库根）执行：**

```bash
python -m src.hydro.train --config config/experiments/hydro_hycom_l0_eos003.yaml

python scripts/hydro_cloud_assessment.py compare --split val \
  --baseline-config config/hydro_hycom_l2.yaml --baseline-ckpt outputs/hydro_l2/best.pt \
  --experiment-config config/experiments/hydro_hycom_l0_eos003.yaml \
  --experiment-ckpt outputs/hydro_l0_eos003/best.pt \
  --stats-npz data/processed/stats/hydro_zscore.npz \
  --out-table-md submission/tables/hydro_l0_eos003_vs_l2_val.md \
  --out-summary-json AutoDL/outputs/cloud/hydro_compare_val_summary_eos003.json

python scripts/hydro_cloud_assessment.py compare --split test \
  --baseline-config config/hydro_hycom_l2.yaml --baseline-ckpt outputs/hydro_l2/best.pt \
  --experiment-config config/experiments/hydro_hycom_l0_eos003.yaml \
  --experiment-ckpt outputs/hydro_l0_eos003/best.pt \
  --stats-npz data/processed/stats/hydro_zscore.npz \
  --out-table-md submission/tables/hydro_l0_eos003_vs_l2_test.md \
  --out-summary-json AutoDL/outputs/cloud/hydro_compare_test_summary_eos003.json
```

`compare` 的 JSON 含 **`summary`（含 `baseline_nrmse_avg` / `experiment_nrmse_avg` 等总体项）** 与 **`per_feature`（含每通道 `baseline_nrmse` / `experiment_nrmse`）**；与 `eos005` 产物字段对齐，便于横比。`--stats-npz` 可选，用于表内物理尺度 RMSE 与 raw 块中的 `rmse_physical_scale`。

---

## 4. 相关索引

- 清单全文：`docs/后续开发工作清单_未完成项与云端L0专项.md`（§2、§3）
- 命令细则：`docs/下一步执行清单_云端评估前端与L0优化.md`
- 云端目录总览：`docs/云端训练与目录归档.md`
