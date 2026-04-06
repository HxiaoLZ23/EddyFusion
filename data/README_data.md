# 数据说明

命题方数据集位于仓库根目录 **`服创数据集/`**（与 `config/data.yaml` 中 `paths.raw_root` 一致）。若需改用 `data/raw/`，请同步修改配置并更新本说明。预处理后产物写入 `data/processed/` 下 `eddy/`、`hydro/`、`anomaly/`。

**参赛方正式说明**：请优先阅读 **`服创数据集/数据集说明.md`**；仓库内摘要见 **`docs/命题方数据集说明.md`**（含海区、划分、指标与输入输出约定）。

## 海域要素预测（HYCOM NetCDF）

- **路径示例**：`服创数据集/海域要素预测/1994/19940101.nc`
- **维度**：`time`×`lat`×`lon` = 12×138×125（示例文件）；`time` 为 `datetime64`。
- **变量（命题方名称）**：**SST、SSS、SSU、SSV**（NetCDF 内多为小写 `sst`/`sss`/`sss` 等，**命名映射见 `config/variable_map.yaml`**）。
- **分辨率**：命题方说明为 **hourly**；**输出为未来 72 小时**（与 `config/hydro_hycom.yaml` 中 `output_steps: 72` 对齐）。
- **指标**：命题方「均方误差」指 **NRMSE**。
- **划分**：训练 **1994～2013**，验证 **2015**，测试 **2014**（见 `config/data.yaml` 的 `hydro_year_split`）；**勿用验证集参与训练**。
- **元数据**：来源 HYCOM GOFS 等，CF 约定以文件为准。

变量名、时空范围与划分方式另见 `config/data.yaml` 与《A09-项目开发文档》第四节；更新时同步修改本说明。
