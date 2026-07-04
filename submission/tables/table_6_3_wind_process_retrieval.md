# 表 6-3  历史风过程检索结果统计

> 由 `scripts/anomaly_wind_process_retrieval_stats.py` 生成。
> 检索评测在 **Oracle 真值时空窗** + 合成区域平均观测风速查询曲线下统计，验收 KB 索引与 DTW 重排链路（非严格物理匹配验证）。

**表 6-3  历史风过程检索结果统计**

| 测试内容 | 数值 |
| --- | --- |
| 历史风过程数量 | **10378** |
| Top-K | **10** |
| DTW 重排完成率 | **100%** |
| 检索成功率 | **100%** |

## 口径说明

- 知识库总事件：**13547**；含非零 ``wind_track_mps`` 过程：**10378**；
  仅峰值常数降级：**30**。
- 评测样本：测试年 [2024]、peak≥34.0 kt 共 **89** 条。
- DTW 口径：``regional_mean_obs_vs_ibtracs_center``（查询异常窗内区域平均 ``wind_obs`` ↔ 历史 ``wind_track_mps``；z-score 后比时间演化形态）。

复现：

```bash
python scripts/anomaly_wind_process_retrieval_stats.py
```