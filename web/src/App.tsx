import { Navigate, Route, Routes } from "react-router-dom";
import { DashboardLayout } from "./layout/DashboardLayout";
import { OfflinePage } from "./routes/OfflinePage";
import { RealtimePage } from "./routes/RealtimePage";

export default function App() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route path="/offline" element={<OfflinePage />} />
        <Route path="/realtime" element={<RealtimePage />} />
        <Route path="/" element={<Navigate to="/offline" replace />} />
      </Route>
    </Routes>
  );
}
