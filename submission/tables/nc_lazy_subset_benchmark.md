# NetCDF 懒加载裁剪压测

> 脚本：`scripts/benchmark_nc_lazy_subset.py` · 指标：耗时(s)、tracemalloc 峰值(MB)

| 操作 | 文件 | 体积(MB) | 耗时(s) | 峰值内存(MB) | 备注 |
|------|------|----------|---------|--------------|------|
| probe_nc_meta_lazy | 0129943eb490_2f1fc15e.nc | 3.36 | 4.2515 | 41.62 | time_len=48 |
| subset_netcdf_lazy | 0129943eb490_2f1fc15e.nc | 3.36 | 1.024 | 1.48 | 子集 {'time': 13, 'lat': 64, 'lon': 64} · 0.9426 MB |
| materialize_all_vars | 0129943eb490_2f1fc15e.nc | 3.36 | 0.9429 | 5.29 | 6 vars · 数组 4.5 MB |
| probe_nc_meta_lazy | 013bcf50fb58_ad35dea2.nc | 384.08 | 2.2576 | 0.22 | time_len=300 |
| subset_netcdf_lazy | 013bcf50fb58_ad35dea2.nc | 384.08 | 14.6891 | 518.9 | 子集 {'time': 76, 'lat': 138, 'lon': 125, 'time_eddy': 300, 'latitude_eddy': 160, 'longitude_eddy': 320} · 311.1677 MB |
| materialize_all_vars | 013bcf50fb58_ad35dea2.nc | 384.08 | 4.472 | 666.25 | 10 vars · 数组 548.97 MB |
| probe_nc_meta_lazy | 02799708b3ab_ad35dea2.nc | 384.08 | 2.06 | 0.21 | time_len=300 |
| subset_netcdf_lazy | 02799708b3ab_ad35dea2.nc | 384.08 | 14.2065 | 518.9 | 子集 {'time': 76, 'lat': 138, 'lon': 125, 'time_eddy': 300, 'latitude_eddy': 160, 'longitude_eddy': 320} · 311.1677 MB |
| materialize_all_vars | 02799708b3ab_ad35dea2.nc | 384.08 | 4.3858 | 666.25 | 10 vars · 数组 548.97 MB |

## 结论（论文 §6 非功能）

- `probe_nc_meta` 仅读元数据与坐标，峰值内存显著低于 `materialize_all_vars`。
- `subset_netcdf` 在懒加载上 `isel/sel` 后写出子集，适合大文件先裁剪再分析。
- 正式演示链路应先 ROI/时间裁剪，再跑涡旋/风浪，避免整库载入。