import { Navigate, Route, Routes } from "react-router-dom";
import { TyphoonKbPage } from "../pages/TyphoonKbPage";

export function RealtimePage() {
  return (
    <Routes>
      <Route path="typhoon-kb" element={<TyphoonKbPage mode="realtime" />} />
      <Route path="*" element={<Navigate to="/monitor?source=realtime" replace />} />
    </Routes>
  );
}
