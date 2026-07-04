# 系统功能测试结果表

| 测试层级 | 测试用例 | 测试结果 | 耗时(s) | 说明 |
| --- | --- | --- | --- | --- |
| test_anomaly_detect | `test_compute_anomaly_assessment_uses_three_sigma_levels` | 通过 | 0.001 | - |
| test_anomaly_detect | `test_rerank_candidates_by_dtw_prefers_closest_curve` | 通过 | 0.001 | - |
| test_anomaly_detect | `test_query_typhoon_events_filters_by_time_and_space` | 通过 | 0.025 | - |
| test_anomaly_detect | `test_run_detect_links_demo_typhoon_event` | 通过 | 0.005 | - |
| test_eddy_physics | `test_relative_vorticity_and_okubo_weiss_are_finite` | 通过 | 0.002 | - |
| test_eddy_physics | `test_build_physics_stacked_hw7_shape_dtype_and_range` | 通过 | 0.013 | - |
| test_eddy_physics | `test_build_physics_stacked_hw6_no_ow_is_six_channels` | 通过 | 0.002 | - |
| test_eddy_physics | `test_ablation_profiles_shapes` | 通过 | 0.004 | - |
| test_eddy_physics | `test_build_physics_stacked_hw8_shape_dtype_and_range` | 通过 | 0.024 | - |
| test_paper_system_api | `test_T1_preprocess_meta_probe` | 通过 | 1.441 | - |
| test_paper_system_api | `test_T2_preprocess_subset_roi` | 通过 | 0.743 | - |
| test_paper_system_api | `test_T3_eddy_preview_frame` | 通过 | 4.610 | - |
| test_paper_system_api | `test_T4_eddy_dual_mp4_staged_job` | 通过 | 4.545 | - |
| test_paper_system_api | `test_T5_windwave_forecast` | 通过 | 1.091 | - |
| test_paper_system_api | `test_T6_report_save_and_history` | 通过 | 1.111 | - |
| test_paper_system_api | `test_T7_async_job_windwave_forecast` | 通过 | 1.624 | - |
| test_paper_system_api | `test_T8_realtime_connector_status` | 通过 | 0.018 | - |
| test_web_api_functional | `test_health_and_eddy_ping_endpoints` | 通过 | 0.138 | - |
| test_web_api_functional | `test_typhoon_kb_status_and_query` | 通过 | 0.019 | - |
| test_web_api_functional | `test_windwave_offline_report_endpoint_uses_demo_netcdf` | 通过 | 1.050 | - |

说明：本表由 `python scripts/run_system_tests.py` 自动生成，测试数据采用本地 demo/合成小数据，用于验证系统核心功能链路是否可在本地稳定运行。
