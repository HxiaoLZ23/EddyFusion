import { NavLink, Outlet } from "react-router-dom";

const linkStyle = ({ isActive }: { isActive: boolean }) => ({
  padding: "0.35rem 0.75rem",
  borderRadius: 6,
  textDecoration: "none",
  color: isActive ? "#fff" : "#0f172a",
  background: isActive ? "#0369a1" : "#e2e8f0",
  fontWeight: isActive ? 600 : 400,
});

export function DashboardLayout() {
  return (
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
        <strong style={{ fontSize: "1.05rem" }}>EddyFusion · 实时 / 离线同屏（规划 §2）</strong>
        <nav style={{ display: "flex", gap: 8 }}>
          <NavLink to="/offline" style={linkStyle}>
            离线系统
          </NavLink>
          <NavLink to="/realtime" style={linkStyle}>
            实时系统
          </NavLink>
        </nav>
        <span style={{ marginLeft: "auto", fontSize: 12, color: "#64748b" }}>
          API: {import.meta.env.VITE_API_BASE || "(未配置 VITE_API_BASE)"}
        </span>
      </header>
      <main style={{ flex: 1, padding: 16 }}>
        <Outlet />
      </main>
    </div>
  );
}
