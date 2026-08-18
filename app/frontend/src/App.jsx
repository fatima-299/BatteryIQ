import { useState, useEffect } from "react";
import FleetDashboard from "./pages/FleetDashboard";
import CellDeepDive from "./pages/CellDeepDive";
import BatteryChat from "./pages/BatteryChat";
import ImageAnalyser from "./pages/ImageAnalyser";
import ReportGenerator from "./pages/ReportGenerator";

const NAV = [
  { id: "fleet",   label: "Fleet Dashboard", icon: "🔋" },
  { id: "cell",    label: "Cell Deep-Dive",  icon: "📊" },
  { id: "chat",    label: "BatteryChat",     icon: "💬" },
  { id: "image",   label: "Image Analyser",  icon: "🔍" },
  { id: "report",  label: "Report Generator",icon: "📄" },
];

export default function App() {
  const [page, setPage]       = useState("fleet");
  const [selectedCell, setSelectedCell] = useState(null);

  const navigateToCell = (cellId) => {
    setSelectedCell(cellId);
    setPage("cell");
  };

  return (
    <div style={{ display:"flex", height:"100vh",
                  fontFamily:"'Segoe UI',sans-serif",
                  background:"#0F172A" }}>

      {/* ── Sidebar ── */}
      <div style={{ width:220, background:"#1E293B",
                    display:"flex", flexDirection:"column",
                    padding:"24px 0", borderRight:"1px solid #334155" }}>

        {/* Logo */}
        <div style={{ padding:"0 20px 24px",
                      borderBottom:"1px solid #334155" }}>
          <div style={{ fontSize:22, fontWeight:800,
                        color:"#38BDF8", letterSpacing:1 }}>
            🔋 BatteryIQ
          </div>
          <div style={{ fontSize:11, color:"#64748B",
                        marginTop:4 }}>
            EV Fleet Intelligence
          </div>
        </div>

        {/* Nav items */}
        <nav style={{ padding:"16px 0", flex:1 }}>
          {NAV.map(n => (
            <div key={n.id}
              onClick={() => setPage(n.id)}
              style={{
                padding:"12px 20px",
                cursor:"pointer",
                display:"flex", alignItems:"center", gap:10,
                fontSize:14, fontWeight: page===n.id ? 600 : 400,
                color: page===n.id ? "#38BDF8" : "#94A3B8",
                background: page===n.id ? "#0F172A" : "transparent",
                borderLeft: page===n.id
                  ? "3px solid #38BDF8" : "3px solid transparent",
                transition:"all 0.2s",
              }}>
              <span style={{ fontSize:18 }}>{n.icon}</span>
              {n.label}
            </div>
          ))}
        </nav>

        {/* Footer */}
        <div style={{ padding:"16px 20px",
                      borderTop:"1px solid #334155",
                      fontSize:11, color:"#475569" }}>
          <div>Physics-Informed ML</div>
          <div style={{ color:"#38BDF8", marginTop:2 }}>v1.0.0</div>
        </div>
      </div>

      {/* ── Main content ── */}
      <div style={{ flex:1, overflow:"auto" }}>
        {page === "fleet"  && <FleetDashboard onCellClick={navigateToCell}/>}
        {page === "cell"   && <CellDeepDive   selectedCell={selectedCell}/>}
        {page === "chat"   && <BatteryChat/>}
        {page === "image"  && <ImageAnalyser/>}
        {page === "report" && <ReportGenerator/>}
      </div>
    </div>
  );
}
