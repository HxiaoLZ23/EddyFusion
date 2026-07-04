import { Navigate, Route, Routes } from "react-router-dom";
import { TyphoonKbPage } from "../pages/TyphoonKbPage";

/** 遗留深链：台风 KB；其余 /offline/* 重定向至监测总览 */
export function OfflinePage() {
  return (
    <Routes>
      <Route path="typhoon-kb" element={<TyphoonKbPage mode="offline" />} />
      <Route path="*" element={<Navigate to="/monitor" replace />} />
    </Routes>
  );
}
