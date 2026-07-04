import { useEffect } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { OceanSessionProvider } from "../dashboard/offlineSession";
import { MonitorPage } from "../pages/MonitorPage";

const linkStyle = ({ isActive }: { isActive: boolean }) => ({
  padding: "0.35rem 0.75rem",
  borderRadius: 6,
  textDecoration: "none",
  color: isActive ? "#fff" : "#0f172a",
  background: isActive ? "#0369a1" : "#e2e8f0",
  fontWeight: isActive ? 600 : 400,
});

/** 论文 §4.5 顶栏四入口 */
const MAIN_NAV = [
  { to: "/monitor", label: "监测总览" },
  { to: "/eddy", label: "涡旋分析" },
  { to: "/windwave", label: "风浪分析" },
  { to: "/reports", label: "报告管理" },
] as const;

function isMonitorRoute(pathname: string): boolean {
  return pathname === "/monitor" || pathname === "/monitor/";
}

export function DashboardLayout() {
  const location = useLocation();
  const onMonitor = isMonitorRoute(location.pathname);

  useEffect(() => {
    document.title = "海洋环境智能监测系统";
  }, []);

  return (
    <OceanSessionProvider>
      <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
        <header
          style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
            padding: "12px 20px",
            background: "#fff",
            borderBottom: "1px solid #cbd5e1",
          }}
        >
          <strong style={{ fontSize: "1.05rem" }}>海洋环境智能监测系统</strong>
          <nav style={{ display: "flex", gap: 8 }}>
            {MAIN_NAV.map(({ to, label }) => (
              <NavLink key={to} to={to} style={linkStyle}>
                {label}
              </NavLink>
            ))}
          </nav>
          <span style={{ marginLeft: "auto", fontSize: 12, color: "#64748b" }}>
            API: {import.meta.env.VITE_API_BASE || "(未配置 VITE_API_BASE)"}
          </span>
        </header>
        <main style={{ flex: 1, padding: 16 }}>
          {/* 监测总览 keep-alive：切到其他顶栏页再返回时不卸载，避免涡旋/风浪重复推理 */}
          <div hidden={!onMonitor} aria-hidden={!onMonitor}>
            <MonitorPage active={onMonitor} />
          </div>
          <div hidden={onMonitor} aria-hidden={onMonitor}>
            <Outlet />
          </div>
        </main>
      </div>
    </OceanSessionProvider>
  );
}
