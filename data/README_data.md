# 数据说明

命题方数据集位于仓库根目录 **`服创数据集/`**（与 `config/data.yaml` 中 `paths.raw_root` 一致）。若需改用 `data/raw/`，请同步修改配置并更新本说明。

**`data/processed/` 各子目录何时会有文件**

| 子目录 | 含义 | 何时非空 |
|--------|------|----------|
| **`hydro/`** | 水文 ConvLSTM 用 npz | 执行 `python -m src.preprocess.hydro_dataset ... --from-nc` 后生成 `X_*.npz` / `y_*.npz`；**仅跑水文训练时只需本目录有数据** |
| `eddy/` | 涡旋等 | 运行对应 `preprocess.eddy_*` 或项目内涡旋预处理脚本后 |
| `anomaly/` | 风-浪异常等 | 运行对应 `preprocess.anomaly_*` 后 |
| `stats/` | 标准化等统计量 | 水文预处理会写入如 `hydro_zscore.npz` |

因此仅做 **HYCOM 水文**时，`config/hydro_hycom.yaml` 只读 **`data/processed/hydro/`**；**`anomaly/`、`eddy/` 为空是正常现象**。

**参赛方正式说明**：请优先阅读 **`服创数据集/数据集说明.md`**；仓库内摘要见 **`docs/命题方数据集说明.md`**（含海区、划分、指标与输入输出约定）。

## 海域要素预测（HYCOM NetCDF）

- **路径示例**：`服创数据集/海域要素预测/1994/19940101.nc`
- **维度**：`time`×`lat`×`lon` = 12×138×125（示例文件）；`time` 为 `datetime64`。
- **变量（命题方名称）**：**SST、SSS、SSU、SSV**。NetCDF 内实际变量名因文件而异，**以 `config/variable_map.yaml` 候选列表为准**（盐度常见为三小写 **s** 连续；东/北向流速对应 `ssu` / 小写 **ssv**）。
- **缺测**：`short` 型常带 `_FillValue`（如 −30000），陆地格点解码后多为 **NaN**；预处理会统计并替换非有限值，Z-score 对 nan 做稳健估计，详见 `docs/实施过程与局限性.md`。
- **分辨率**：命题方说明为 **hourly**；**输出为未来 72 小时**（与 `config/hydro_hycom.yaml` 中 `output_steps: 72` 对齐）。
- **指标**：命题方「均方误差」指 **NRMSE**。
- **划分**：训练 **1994～2013**，验证 **2015**，测试 **2014**（见 `config/data.yaml` 的 `hydro_year_split`）；**勿用验证集参与训练**。
- **元数据**：来源 HYCOM GOFS 等，CF 约定以文件为准。

变量名、时空范围与划分方式另见 `config/data.yaml` 与《A09-项目开发文档》第四节；更新时同步修改本说明。
