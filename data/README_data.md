# 数据说明

命题方数据集位于仓库根目录 **`服创数据集/`**（与 `config/data.yaml` 中 `paths.raw_root` 一致）。若需改用 `data/raw/`，请同步修改配置并更新本说明。预处理后产物写入 `data/processed/` 下 `eddy/`、`hydro/`、`anomaly/`。

## 海域要素预测（HYCOM NetCDF）

- **路径示例**：`服创数据集/海域要素预测/1994/19940101.nc`
- **维度**：`time`×`lat`×`lon` = 12×138×125（示例文件）；`time` 为 `datetime64`。
- **变量（float32）**：`sst`、`sss`、`ssu`、`ssv`（与 T/S/u/v 对应；**命名映射见 `config/variable_map.yaml`**）。
- **元数据**：来源 HYCOM（NRL），CF-1.6。
- **风速/波高**：当前单日文件中**未包含**；需与「风-浪异常识别」数据或其它再分析时空对齐后合并，或水文基线先使用 **四要素**。

变量名、时空范围与划分方式另见 `config/data.yaml` 与《A09-项目开发文档》第四节；更新时同步修改本说明。
