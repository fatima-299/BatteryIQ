import { useState, useEffect } from "react";

const API = "http://localhost:8000";

export default function CellDeepDive({ selectedCell }) {
  const [cellId,   setCellId]   = useState(selectedCell || "");
  const [data,     setData]     = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState("");

  const load = (id) => {
    if (!id) return;
    setLoading(true); setError("");
    fetch(`${API}/fleet/${id}`)
      .then(r => r.ok ? r.json() : Promise.reject("Not found"))
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  };

  useEffect(() => { if (selectedCell) load(selectedCell); }, [selectedCell]);

  const SOHColor = (soh) =>
    soh >= 90 ? "#1D9E75" : soh >= 80 ? "#EF9F27" : "#EF4444";

  // Simple SVG line chart
  const SOHChart = ({ history }) => {
    if (!history?.length) return null;
    const W=600, H=200, PAD=40;
    const maxC = Math.max(...history.map(h=>h.cycle_number));
    const minS = Math.min(...history.map(h=>h.soh_pct), 75);
    const maxS = Math.max(...history.map(h=>h.soh_pct), 105);
    const cx = c => PAD + (c/maxC)*(W-2*PAD);
    const cy = s => H-PAD - ((s-minS)/(maxS-minS))*(H-2*PAD);

    const pts = history.map(h=>
      `${cx(h.cycle_number)},${cy(h.soh_pct)}`).join(" ");
    const eolY = cy(80);

    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",height:220}}>
        {/* Grid lines */}
        {[80,85,90,95,100].map(v=>(
          <line key={v} x1={PAD} y1={cy(v)}
            x2={W-PAD} y2={cy(v)}
            stroke="#334155" strokeDasharray="4"/>
        ))}
        {/* EOL line */}
        <line x1={PAD} y1={eolY} x2={W-PAD} y2={eolY}
          stroke="#EF4444" strokeDasharray="6" strokeWidth={1.5}/>
        <text x={W-PAD+4} y={eolY+4} fill="#EF4444"
          fontSize={10}>EOL 80%</text>
        {/* SOH line */}
        <polyline points={pts} fill="none"
          stroke="#38BDF8" strokeWidth={2}/>
        {/* Fill under */}
        <polygon
          points={`${PAD},${H-PAD} ${pts} ${cx(maxC)},${H-PAD}`}
          fill="#38BDF8" opacity={0.08}/>
        {/* Axes */}
        <line x1={PAD} y1={PAD} x2={PAD} y2={H-PAD}
          stroke="#475569"/>
        <line x1={PAD} y1={H-PAD} x2={W-PAD} y2={H-PAD}
          stroke="#475569"/>
        {/* Y labels */}
        {[80,90,100].map(v=>(
          <text key={v} x={PAD-6} y={cy(v)+4}
            fill="#64748B" fontSize={10}
            textAnchor="end">{v}%</text>
        ))}
        <text x={W/2} y={H} fill="#64748B"
          fontSize={11} textAnchor="middle">Cycle Number</text>
        <text x={8} y={H/2} fill="#64748B"
          fontSize={11} textAnchor="middle"
          transform={`rotate(-90,8,${H/2})`}>SOH %</text>
      </svg>
    );
  };

  return (
    <div style={{ padding:32, color:"#E2E8F0" }}>
      <h1 style={{ fontSize:26, fontWeight:800,
                   color:"#F1F5F9", margin:0,
                   marginBottom:8 }}>
        Cell Deep-Dive
      </h1>
      <p style={{ color:"#64748B", fontSize:14,
                  marginBottom:24 }}>
        Select any battery cell to view full degradation history
      </p>

      {/* Search bar */}
      <div style={{ display:"flex", gap:12,
                    marginBottom:28 }}>
        <input
          value={cellId}
          onChange={e=>setCellId(e.target.value)}
          placeholder="Enter cell ID (e.g. B0005, CS2_9, 2017-05-12_c01)"
          style={{ padding:"12px 16px", borderRadius:8,
                   border:"1px solid #334155",
                   background:"#1E293B", color:"#E2E8F0",
                   fontSize:14, width:400, outline:"none" }}
          onKeyDown={e=>e.key==="Enter"&&load(cellId)}
        />
        <button onClick={()=>load(cellId)}
          style={{ padding:"12px 24px", borderRadius:8,
                   background:"#0F4C81",
                   border:"1px solid #38BDF8",
                   color:"#38BDF8", cursor:"pointer",
                   fontSize:14, fontWeight:600 }}>
          Load Cell
        </button>
      </div>

      {loading && <div style={{color:"#38BDF8"}}>
        Loading cell data...</div>}
      {error   && <div style={{color:"#EF4444"}}>
        ❌ {error}</div>}

      {data && (
        <div>
          {/* KPI row */}
          <div style={{ display:"flex", gap:16,
                        marginBottom:24 }}>
            {[
              { label:"Current SOH",
                value:`${Number(data.current_soh).toFixed(1)}%`,
                color: SOHColor(data.current_soh) },
              { label:"Risk Score",
                value: Number(data.risk_score).toFixed(1),
                color:"#EF9F27" },
              { label:"Total Cycles",
                value: data.total_cycles,
                color:"#38BDF8" },
              { label:"Alert Status",
                value: data.alert_flag?.replace("_"," "),
                color: data.alert_flag==="EOL_REACHED"?"#EF4444"
                  : data.alert_flag==="WARNING"?"#EF9F27":"#1D9E75"},
              { label:"Chemistry",
                value: data.chemistry,
                color:"#A78BFA" },
              { label:"Source",
                value: data.source,
                color:"#38BDF8" },
            ].map(k=>(
              <div key={k.label}
                style={{ background:"#1E293B", borderRadius:10,
                         padding:"16px 20px", flex:1,
                         border:"1px solid #334155" }}>
                <div style={{ fontSize:12, color:"#64748B",
                              marginBottom:6 }}>{k.label}</div>
                <div style={{ fontSize:22, fontWeight:800,
                              color:k.color }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* SOH Chart */}
          <div style={{ background:"#1E293B", borderRadius:12,
                        padding:24, border:"1px solid #334155",
                        marginBottom:24 }}>
            <h3 style={{ margin:"0 0 16px",
                         color:"#F1F5F9", fontSize:16 }}>
              SOH Degradation Trajectory — {data.cell_id}
            </h3>
            <SOHChart history={data.history}/>
          </div>

          {/* History table */}
          <div style={{ background:"#1E293B", borderRadius:12,
                        border:"1px solid #334155",
                        overflow:"hidden" }}>
            <div style={{ padding:"16px 20px",
                          borderBottom:"1px solid #334155",
                          display:"flex",
                          justifyContent:"space-between",
                          alignItems:"center" }}>
              <h3 style={{ margin:0, fontSize:16,
                           color:"#F1F5F9" }}>
                Cycle History
              </h3>
              <span style={{ fontSize:12, color:"#64748B" }}>
                {data.history?.length} cycles
              </span>
            </div>
            <div style={{ maxHeight:300, overflowY:"auto" }}>
              <table style={{ width:"100%",
                              borderCollapse:"collapse" }}>
                <thead style={{ position:"sticky", top:0 }}>
                  <tr style={{ background:"#0F172A" }}>
                    {["Cycle","SOH %","Capacity (Ah)",
                      "Resistance (Ω)","Risk Score","Status"]
                      .map(h=>(
                      <th key={h}
                        style={{ padding:"10px 16px",
                                 textAlign:"left",
                                 fontSize:12, color:"#64748B",
                                 fontWeight:600 }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(data.history||[]).slice(-50).reverse()
                    .map((h,i)=>(
                    <tr key={i}
                      style={{ borderBottom:"1px solid #1E293B",
                               background:i%2===0
                                 ?"#1E293B":"#162032" }}>
                      <td style={{ padding:"8px 16px",
                                   color:"#94A3B8",
                                   fontSize:13 }}>
                        {h.cycle_number}
                      </td>
                      <td style={{ padding:"8px 16px",
                                   fontWeight:700,
                                   color: SOHColor(h.soh_pct),
                                   fontSize:13 }}>
                        {Number(h.soh_pct||0).toFixed(2)}%
                      </td>
                      <td style={{ padding:"8px 16px",
                                   color:"#94A3B8",
                                   fontSize:13 }}>
                        {Number(h.cycle_capacity_ah||0).toFixed(3)}
                      </td>
                      <td style={{ padding:"8px 16px",
                                   color:"#94A3B8",
                                   fontSize:13 }}>
                        {h.internal_resistance
                          ? Number(h.internal_resistance).toFixed(4)
                          : "N/A"}
                      </td>
                      <td style={{ padding:"8px 16px",
                                   color:"#EF9F27",
                                   fontSize:13 }}>
                        {Number(h.risk_score||0).toFixed(1)}
                      </td>
                      <td style={{ padding:"8px 16px" }}>
                        <span style={{
                          fontSize:11, padding:"2px 8px",
                          borderRadius:20,
                          background: h.alert_flag==="EOL_REACHED"
                            ?"#EF444422":"#1D9E7522",
                          color: h.alert_flag==="EOL_REACHED"
                            ?"#EF4444":"#1D9E75",
                        }}>
                          {h.alert_flag?.replace("_"," ")}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
