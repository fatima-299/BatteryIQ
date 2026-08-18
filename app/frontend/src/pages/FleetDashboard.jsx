import { useState, useEffect } from "react";

const API = "http://localhost:8000";

const COLOR = {
  excellent: "#1D9E75", good: "#38BDF8",
  fair: "#EF9F27", poor: "#F97316", critical: "#EF4444"
};

const ALERT_COLOR = {
  OK: "#1D9E75", MONITOR: "#38BDF8",
  WARNING: "#EF9F27", EOL_REACHED: "#EF4444"
};

function KPICard({ label, value, sub, color="#38BDF8" }) {
  return (
    <div style={{ background:"#1E293B", borderRadius:12,
                  padding:"20px 24px", flex:1,
                  border:"1px solid #334155" }}>
      <div style={{ fontSize:13, color:"#64748B",
                    marginBottom:8 }}>{label}</div>
      <div style={{ fontSize:32, fontWeight:800,
                    color, marginBottom:4 }}>{value}</div>
      {sub && <div style={{ fontSize:12,
                            color:"#475569" }}>{sub}</div>}
    </div>
  );
}

export default function FleetDashboard({ onCellClick }) {
  const [fleet,   setFleet]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [search,  setSearch]  = useState("");
  const [filter,  setFilter]  = useState("all");
  const [sortBy,  setSortBy]  = useState("risk_score");

  useEffect(() => {
    fetch(`${API}/fleet/`)
      .then(r => r.json())
      .then(d => { setFleet(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div style={{ display:"flex", alignItems:"center",
                  justifyContent:"center", height:"100%",
                  color:"#38BDF8", fontSize:18 }}>
      Loading fleet data...
    </div>
  );

  if (!fleet) return (
    <div style={{ padding:40, color:"#EF4444" }}>
      ❌ Cannot connect to API. Make sure FastAPI is running on port 8000.
    </div>
  );

  const cells = (fleet.cells || [])
    .filter(c => {
      const matchSearch = !search ||
        c.cell_id?.toLowerCase().includes(search.toLowerCase()) ||
        c.source?.toLowerCase().includes(search.toLowerCase());
      const matchFilter = filter === "all" ||
        c.alert_flag === filter;
      return matchSearch && matchFilter;
    })
    .sort((a,b) => {
      if (sortBy === "risk_score")
        return (b.risk_score||0) - (a.risk_score||0);
      if (sortBy === "soh_pct")
        return (a.soh_pct||0) - (b.soh_pct||0);
      return (a.cell_id||"").localeCompare(b.cell_id||"");
    });

  const eol     = cells.filter(c=>c.alert_flag==="EOL_REACHED").length;
  const warning = cells.filter(c=>c.alert_flag==="WARNING").length;
  const monitor = cells.filter(c=>c.alert_flag==="MONITOR").length;
  const avgSOH  = fleet.avg_soh || 0;

  return (
    <div style={{ padding:32, color:"#E2E8F0" }}>

      {/* Header */}
      <div style={{ marginBottom:28 }}>
        <h1 style={{ fontSize:26, fontWeight:800,
                     color:"#F1F5F9", margin:0 }}>
          Fleet Overview
        </h1>
        <p style={{ color:"#64748B", marginTop:6, fontSize:14 }}>
          {fleet.total_cells} cells monitored across
          NASA, Stanford & CALCE datasets
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display:"flex", gap:16, marginBottom:28 }}>
        <KPICard label="Average Fleet SOH"
          value={`${avgSOH}%`}
          sub="Across all cells"
          color="#38BDF8"/>
        <KPICard label="Total Cells"
          value={fleet.total_cells}
          sub="189 cells monitored"
          color="#1D9E75"/>
        <KPICard label="Cells at EOL"
          value={eol}
          sub="SOH < 80% — replace now"
          color="#EF4444"/>
        <KPICard label="Cells Warning"
          value={warning}
          sub="SOH 80-85% — monitor"
          color="#EF9F27"/>
        <KPICard label="Cells Monitor"
          value={monitor}
          sub="SOH 85-90% — watch"
          color="#38BDF8"/>
      </div>

      {/* Filters */}
      <div style={{ display:"flex", gap:12,
                    marginBottom:20, alignItems:"center",
                    flexWrap:"wrap" }}>
        <input
          placeholder="🔍 Search cell ID or source..."
          value={search}
          onChange={e=>setSearch(e.target.value)}
          style={{ padding:"10px 16px", borderRadius:8,
                   border:"1px solid #334155",
                   background:"#1E293B", color:"#E2E8F0",
                   fontSize:14, width:260, outline:"none" }}
        />
        {["all","EOL_REACHED","WARNING","MONITOR","OK"].map(f=>(
          <button key={f}
            onClick={()=>setFilter(f)}
            style={{
              padding:"8px 16px", borderRadius:8,
              border:`1px solid ${filter===f?"#38BDF8":"#334155"}`,
              background: filter===f?"#0F4C81":"#1E293B",
              color: filter===f?"#38BDF8":"#94A3B8",
              cursor:"pointer", fontSize:13, fontWeight:500,
            }}>
            {f==="all"?"All":f.replace("_"," ")}
          </button>
        ))}
        <select
          value={sortBy}
          onChange={e=>setSortBy(e.target.value)}
          style={{ padding:"8px 16px", borderRadius:8,
                   border:"1px solid #334155",
                   background:"#1E293B", color:"#E2E8F0",
                   fontSize:13, cursor:"pointer" }}>
          <option value="risk_score">Sort: Risk Score ↓</option>
          <option value="soh_pct">Sort: SOH ↑</option>
          <option value="cell_id">Sort: Cell ID</option>
        </select>
      </div>

      {/* Cell Table */}
      <div style={{ background:"#1E293B", borderRadius:12,
                    border:"1px solid #334155", overflow:"hidden" }}>
        <table style={{ width:"100%",
                        borderCollapse:"collapse" }}>
          <thead>
            <tr style={{ background:"#0F172A",
                         borderBottom:"1px solid #334155" }}>
              {["Cell ID","Source","Chemistry","SOH %",
                "Risk Score","Alert","Status","Cycles","Action"]
                .map(h=>(
                <th key={h} style={{ padding:"12px 16px",
                  textAlign:"left", fontSize:12,
                  color:"#64748B", fontWeight:600,
                  textTransform:"uppercase",
                  letterSpacing:"0.5px" }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cells.map((c,i)=>(
              <tr key={c.cell_id}
                style={{
                  borderBottom:"1px solid #1E293B",
                  background: i%2===0?"#1E293B":"#162032",
                  cursor:"pointer",
                  transition:"background 0.15s",
                }}
                onMouseEnter={e=>
                  e.currentTarget.style.background="#0F4C81"}
                onMouseLeave={e=>
                  e.currentTarget.style.background=
                    i%2===0?"#1E293B":"#162032"}
                onClick={()=>onCellClick(c.cell_id)}>

                <td style={{ padding:"12px 16px",
                             fontWeight:600,
                             color:"#38BDF8",
                             fontSize:13 }}>
                  {c.cell_id}
                </td>
                <td style={{ padding:"12px 16px",
                             color:"#94A3B8",
                             fontSize:13 }}>
                  {c.source}
                </td>
                <td style={{ padding:"12px 16px",
                             color:"#94A3B8",
                             fontSize:13 }}>
                  {c.chemistry}
                </td>
                <td style={{ padding:"12px 16px",
                             fontWeight:700,
                             color: c.soh_pct >= 90?"#1D9E75"
                               : c.soh_pct >= 80?"#EF9F27":"#EF4444",
                             fontSize:14 }}>
                  {Number(c.soh_pct||0).toFixed(1)}%
                </td>
                <td style={{ padding:"12px 16px",
                             color:"#E2E8F0",
                             fontSize:13 }}>
                  {Number(c.risk_score||0).toFixed(1)}
                </td>
                <td style={{ padding:"12px 16px" }}>
                  <span style={{
                    padding:"3px 10px", borderRadius:20,
                    fontSize:11, fontWeight:600,
                    background: ALERT_COLOR[c.alert_flag]+"22",
                    color: ALERT_COLOR[c.alert_flag] || "#94A3B8",
                    border:`1px solid ${ALERT_COLOR[c.alert_flag]||"#334155"}`,
                  }}>
                    {c.alert_flag?.replace("_"," ")}
                  </span>
                </td>
                <td style={{ padding:"12px 16px" }}>
                  <span style={{
                    padding:"3px 10px", borderRadius:20,
                    fontSize:11, fontWeight:600,
                    background: COLOR[c.degradation_category]+"22",
                    color: COLOR[c.degradation_category]||"#94A3B8",
                  }}>
                    {c.degradation_category}
                  </span>
                </td>
                <td style={{ padding:"12px 16px",
                             color:"#94A3B8", fontSize:13 }}>
                  {c.cycle_number}
                </td>
                <td style={{ padding:"12px 16px" }}>
                  <button
                    onClick={e=>{e.stopPropagation();
                      onCellClick(c.cell_id);}}
                    style={{
                      padding:"6px 14px", borderRadius:6,
                      background:"#0F4C81",
                      border:"1px solid #38BDF8",
                      color:"#38BDF8", cursor:"pointer",
                      fontSize:12, fontWeight:600,
                    }}>
                    Deep Dive →
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ padding:"12px 16px",
                      color:"#475569", fontSize:12,
                      borderTop:"1px solid #334155" }}>
          Showing {cells.length} of {fleet.total_cells} cells
        </div>
      </div>
    </div>
  );
}
