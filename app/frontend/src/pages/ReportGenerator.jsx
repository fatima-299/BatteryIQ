import { useState, useEffect } from "react";

const API = "http://localhost:8000";

export default function ReportGenerator() {
  const [cells,      setCells]      = useState([]);
  const [cellId,     setCellId]     = useState("");
  const [reportText, setReportText] = useState("");
  const [nlpResult,  setNlpResult]  = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [error,      setError]      = useState("");

  useEffect(() => {
    fetch(`${API}/chat/cells`)
      .then(r=>r.json())
      .then(d=>setCells(d))
      .catch(()=>{});
  }, []);

  const analyseReport = async () => {
    if (!reportText.trim()) return;
    setLoading(true); setError(""); setNlpResult(null);
    try {
      const form = new FormData();
      form.append("text", reportText);
      const res  = await fetch(`${API}/analyse-report/`,
        { method:"POST", body:form });
      const data = await res.json();
      setNlpResult(data);
      if (data.cell_ids_mentioned?.[0] && !cellId)
        setCellId(data.cell_ids_mentioned[0]);
    } catch {
      setError("Cannot connect to API.");
    }
    setLoading(false);
  };

  const generatePDF = async () => {
    if (!cellId) return;
    setPdfLoading(true); setError("");
    try {
      const res = await fetch(
        `${API}/generate-report/${cellId}`,
        { method:"POST" }
      );
      if (!res.ok) throw new Error("Failed");
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href     = url;
      a.download = `BatteryIQ_${cellId}_report.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError("PDF generation failed. Check OpenAI key.");
    }
    setPdfLoading(false);
  };

  const URGENCY_COLOR = {
    critical:"#EF4444", warning:"#EF9F27",
    normal:"#1D9E75", unknown:"#64748B"
  };

  return (
    <div style={{ padding:32, color:"#E2E8F0" }}>
      <h1 style={{ fontSize:26, fontWeight:800,
                   color:"#F1F5F9", margin:0,
                   marginBottom:8 }}>
        📄 Report Generator
      </h1>
      <p style={{ color:"#64748B", fontSize:14,
                  marginBottom:28 }}>
        Analyse maintenance logs with NLP and generate
        AI-powered PDF health reports
      </p>

      <div style={{ display:"flex", gap:24,
                    flexWrap:"wrap" }}>

        {/* Left: NLP analyser */}
        <div style={{ flex:1, minWidth:340 }}>
          <div style={{ background:"#1E293B", borderRadius:12,
                        padding:24, border:"1px solid #334155",
                        marginBottom:20 }}>
            <h3 style={{ margin:"0 0 16px",
                         color:"#F1F5F9", fontSize:16 }}>
              🧠 NLP Maintenance Log Analyser
            </h3>
            <textarea
              value={reportText}
              onChange={e=>setReportText(e.target.value)}
              placeholder={`Paste maintenance log here...\n\nExample:\nCell B0043 showed critical voltage drop at cycle 10.\nTemperature exceeded 45°C during fast charging.\nUnusual swelling detected on battery casing.\nImmediate inspection recommended.`}
              style={{ width:"100%", height:200,
                       padding:"12px", borderRadius:8,
                       border:"1px solid #334155",
                       background:"#0F172A", color:"#E2E8F0",
                       fontSize:13, lineHeight:1.6,
                       resize:"vertical", outline:"none",
                       boxSizing:"border-box" }}
            />
            <button onClick={analyseReport} disabled={loading}
              style={{ width:"100%", marginTop:12,
                       padding:"12px", borderRadius:8,
                       background:"#0F4C81",
                       border:"1px solid #38BDF8",
                       color:"#38BDF8", cursor:"pointer",
                       fontSize:14, fontWeight:600 }}>
              {loading ? "Analysing..." : "🔍 Analyse Report"}
            </button>
          </div>

          {/* NLP Results */}
          {nlpResult && (
            <div style={{ background:"#1E293B", borderRadius:12,
                          padding:24, border:"1px solid #334155" }}>
              <h3 style={{ margin:"0 0 16px",
                           color:"#F1F5F9", fontSize:16 }}>
                Analysis Results
              </h3>

              {/* Urgency */}
              <div style={{ marginBottom:16 }}>
                <div style={{ fontSize:12, color:"#64748B",
                              marginBottom:6 }}>Urgency Level</div>
                <span style={{
                  padding:"6px 16px", borderRadius:20,
                  background: URGENCY_COLOR[nlpResult.urgency_level]+"22",
                  color: URGENCY_COLOR[nlpResult.urgency_level],
                  border:`1px solid ${URGENCY_COLOR[nlpResult.urgency_level]}`,
                  fontSize:13, fontWeight:700,
                  textTransform:"uppercase",
                }}>
                  {nlpResult.urgency_level}
                </span>
              </div>

              {/* Cells mentioned */}
              {nlpResult.cell_ids_mentioned?.length > 0 && (
                <div style={{ marginBottom:16 }}>
                  <div style={{ fontSize:12, color:"#64748B",
                                marginBottom:6 }}>
                    Cells Mentioned
                  </div>
                  <div style={{ display:"flex", gap:8,
                                flexWrap:"wrap" }}>
                    {nlpResult.cell_ids_mentioned.map(c=>(
                      <span key={c} style={{
                        padding:"4px 12px", borderRadius:20,
                        background:"#38BDF822",
                        color:"#38BDF8",
                        border:"1px solid #38BDF8",
                        fontSize:13, fontWeight:600,
                        cursor:"pointer",
                      }}
                      onClick={()=>setCellId(c)}>
                        {c}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Anomalies */}
              {nlpResult.anomalies_detected?.length > 0 && (
                <div style={{ marginBottom:16 }}>
                  <div style={{ fontSize:12, color:"#64748B",
                                marginBottom:6 }}>
                    Anomalies Detected
                  </div>
                  <div style={{ display:"flex", gap:8,
                                flexWrap:"wrap" }}>
                    {nlpResult.anomalies_detected.map(a=>(
                      <span key={a} style={{
                        padding:"4px 12px", borderRadius:20,
                        background:"#EF444422", color:"#EF4444",
                        border:"1px solid #EF4444",
                        fontSize:13,
                      }}>⚠️ {a}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Summary */}
              <div style={{ background:"#0F172A",
                            borderRadius:8, padding:14,
                            marginBottom:16,
                            fontSize:13, color:"#CBD5E1",
                            lineHeight:1.6 }}>
                {nlpResult.summary}
              </div>

              {/* Recommendations */}
              {nlpResult.recommendations?.length > 0 && (
                <div>
                  <div style={{ fontSize:12, color:"#64748B",
                                marginBottom:8 }}>
                    Recommendations
                  </div>
                  {nlpResult.recommendations.map((r,i)=>(
                    <div key={i} style={{ display:"flex",
                                          gap:8, marginBottom:6,
                                          fontSize:13,
                                          color:"#94A3B8" }}>
                      <span style={{ color:"#38BDF8" }}>→</span>
                      {r}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: PDF Generator */}
        <div style={{ flex:1, minWidth:300 }}>
          <div style={{ background:"#1E293B", borderRadius:12,
                        padding:24, border:"1px solid #334155",
                        marginBottom:20 }}>
            <h3 style={{ margin:"0 0 16px",
                         color:"#F1F5F9", fontSize:16 }}>
              📊 AI PDF Report Generator
            </h3>
            <p style={{ fontSize:13, color:"#64748B",
                        marginBottom:20, lineHeight:1.6 }}>
              Select a cell to generate a complete PDF report
              with SOH chart, metrics table, and
              GPT-4o written narrative.
            </p>

            <label style={{ fontSize:13, color:"#94A3B8",
                            marginBottom:6, display:"block" }}>
              Select Cell:
            </label>
            <select
              value={cellId}
              onChange={e=>setCellId(e.target.value)}
              style={{ width:"100%", padding:"10px 14px",
                       borderRadius:8,
                       border:"1px solid #334155",
                       background:"#0F172A", color:"#E2E8F0",
                       fontSize:14, cursor:"pointer",
                       marginBottom:16 }}>
              <option value="">Choose a cell...</option>
              {cells.map(c=>(
                <option key={c.cell_id} value={c.cell_id}>
                  {c.cell_id} — {c.chemistry} ({c.source})
                </option>
              ))}
            </select>

            <button onClick={generatePDF}
              disabled={!cellId || pdfLoading}
              style={{ width:"100%", padding:"14px",
                       borderRadius:8, background: !cellId
                         ?"#1E293B":"#0F4C81",
                       border:`1px solid ${!cellId
                         ?"#334155":"#38BDF8"}`,
                       color: !cellId?"#475569":"#38BDF8",
                       cursor: !cellId?"not-allowed":"pointer",
                       fontSize:15, fontWeight:700 }}>
              {pdfLoading
                ? "⏳ Generating PDF..."
                : "📥 Download PDF Report"}
            </button>

            {error && <div style={{ color:"#EF4444",
                                    marginTop:12,
                                    fontSize:13 }}>
              ❌ {error}</div>}
          </div>

          {/* What the PDF contains */}
          <div style={{ background:"#1E293B", borderRadius:12,
                        padding:24, border:"1px solid #334155" }}>
            <h4 style={{ margin:"0 0 16px",
                         color:"#F1F5F9", fontSize:14 }}>
              📋 Report Contents
            </h4>
            {[
              "Cell identification and metadata",
              "KPI table (SOH, risk, cycles, status)",
              "SOH degradation chart (last 200 cycles)",
              "GPT-4o written health assessment",
              "Degradation analysis paragraph",
              "Maintenance recommendations",
            ].map((item, i) => (
              <div key={i} style={{ display:"flex", gap:10,
                                    marginBottom:10,
                                    fontSize:13,
                                    color:"#94A3B8" }}>
                <span style={{ color:"#1D9E75",
                               fontWeight:700 }}>✓</span>
                {item}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
