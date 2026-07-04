import { Navigate, Route, Routes } from "react-router-dom";
import { DashboardLayout } from "./layout/DashboardLayout";
import { EddyAnalysisPage } from "./pages/EddyAnalysisPage";
import { ReportsPage } from "./pages/ReportsPage";
import { WindwaveAnalysisPage } from "./pages/WindwaveAnalysisPage";
import { OfflinePage } from "./routes/OfflinePage";
import { RealtimePage } from "./routes/RealtimePage";

export default function App() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route path="/monitor" element={null} />
        <Route path="/eddy" element={<EddyAnalysisPage />} />
        <Route path="/windwave" element={<WindwaveAnalysisPage />} />
        <Route path="/reports" element={<ReportsPage />} />
        <Route path="/offline/*" element={<OfflinePage />} />
        <Route path="/realtime/*" element={<RealtimePage />} />
        <Route path="/" element={<Navigate to="/monitor" replace />} />
      </Route>
    </Routes>
  );
}
