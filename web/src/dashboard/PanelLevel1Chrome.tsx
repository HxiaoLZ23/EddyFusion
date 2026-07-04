import type { ReactNode } from "react";

type Props = {
  children: ReactNode;
};

/** 监测总览同屏模块布局容器（原 L1 角标入口已移除，专页请走顶栏 /eddy、/windwave）。 */
export function PanelLevel1Chrome({ children }: Props) {
  return <div className="ocean-dashboard__l1-chrome">{children}</div>;
}
