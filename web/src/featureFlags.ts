/**
 * 论文系统对齐演示路径：强制关闭水文大屏（论文全文无水文模块）。
 * 设为 false 且 VITE_SHOW_HYDRO=true 时可恢复仓内水文 UI。
 */
export const PAPER_ALIGN_UI = true;

/**
 * @see docs/实验与结果归档/水文_其他指标与能用标准归档.md §5
 */
export const SHOW_HYDRO_UI =
  !PAPER_ALIGN_UI && import.meta.env.VITE_SHOW_HYDRO === "true";
