import { useState, useRef } from "react";

const API = "http://localhost:8000";

export default function ImageAnalyser() {
  const [file,    setFile]    = useState(null);
  const [preview, setPreview] = useState(null);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");
  const inputRef = useRef();

  const handleFile = (f) => {
    setFile(f);
    setResult(null);
    setError("");
    const reader = new FileReader();
    reader.onload = e => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const analyse = async () => {
    if (!file) return;
    setLoading(true); setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const res  = await fetch(`${API}/analyse-image/`,
        { method:"POST", body:form });
      const data = await res.json();
      setResult(data);
    } catch {
      setError("Cannot connect to API. Make sure FastAPI is running.");
    }
    setLoading(false);
  };

  const SEVERITY_COLOR = {
    none:"#1D9E75", mild:"#38BDF8",
    moderate:"#EF9F27", severe:"#EF4444"
  };

  return (
    <div style={{ padding:32, color:"#E2E8F0" }}>
      <h1 style={{ fontSize:26, fontWeight:800,
                   color:"#F1F5F9", margin:0,
                   marginBottom:8 }}>
        🔍 Battery Image Analyser
      </h1>
      <p style={{ color:"#64748B", fontSize:14,
                  marginBottom:28 }}>
        Upload a battery image to detect swelling, corrosion,
        SEI buildup, and thermal damage using OpenCV
      </p>

      <div style={{ display:"flex", gap:24,
                    flexWrap:"wrap" }}>

        {/* Upload panel */}
        <div style={{ flex:1, minWidth:300 }}>
          <div
            onClick={()=>inputRef.current.click()}
            onDrop={e=>{e.preventDefault();
              handleFile(e.dataTransfer.files[0]);}}
            onDragOver={e=>e.preventDefault()}
            style={{
              border:"2px dashed #334155", borderRadius:12,
              padding:40, textAlign:"center",
              cursor:"pointer", background:"#1E293B",
              transition:"border-color 0.2s",
              marginBottom:16,
            }}
            onMouseEnter={e=>
              e.currentTarget.style.borderColor="#38BDF8"}
            onMouseLeave={e=>
              e.currentTarget.style.borderColor="#334155"}>
            <div style={{ fontSize:48, marginBottom:12 }}>📷</div>
            <div style={{ fontSize:16, color:"#94A3B8",
                          marginBottom:8 }}>
              Drop battery image here or click to browse
            </div>
            <div style={{ fontSize:12, color:"#475569" }}>
              JPG, PNG, TIFF supported
            </div>
          </div>
          <input ref={inputRef} type="file"
            accept="image/*" style={{ display:"none" }}
            onChange={e=>handleFile(e.target.files[0])}/>

          {preview && (
            <div style={{ marginBottom:16 }}>
              <img src={preview} alt="preview"
                style={{ width:"100%", borderRadius:8,
                         border:"1px solid #334155",
                         maxHeight:250, objectFit:"contain",
                         background:"#0F172A" }}/>
            </div>
          )}

          {file && (
            <button onClick={analyse} disabled={loading}
              style={{ width:"100%", padding:"14px",
                       borderRadius:8, background:"#0F4C81",
                       border:"1px solid #38BDF8",
                       color:"#38BDF8", cursor:"pointer",
                       fontSize:15, fontWeight:700 }}>
              {loading ? "Analysing..." : "🔍 Analyse Image"}
            </button>
          )}
          {error && <div style={{ color:"#EF4444",
                                  marginTop:12, fontSize:13 }}>
            ❌ {error}</div>}
        </div>

        {/* Results panel */}
        <div style={{ flex:1, minWidth:300 }}>
          {!result && !loading && (
            <div style={{ background:"#1E293B",
                          borderRadius:12, padding:32,
                          border:"1px solid #334155",
                          textAlign:"center",
                          color:"#475569" }}>
              <div style={{ fontSize:48,
                            marginBottom:16 }}>🔬</div>
              <div>Upload and analyse an image to see
                defect detection results</div>
            </div>
          )}

          {loading && (
            <div style={{ background:"#1E293B",
                          borderRadius:12, padding:32,
                          border:"1px solid #334155",
                          textAlign:"center",
                          color:"#38BDF8" }}>
              <div style={{ fontSize:32,
                            marginBottom:12 }}>⚙️</div>
              <div>Running OpenCV analysis...</div>
            </div>
          )}

          {result && (
            <div style={{ display:"flex",
                          flexDirection:"column", gap:16 }}>

              {/* Severity badge */}
              <div style={{ background:"#1E293B",
                            borderRadius:12, padding:20,
                            border:`2px solid ${
                              SEVERITY_COLOR[result.severity]}` }}>
                <div style={{ fontSize:13, color:"#64748B",
                              marginBottom:8 }}>
                  Overall Severity
                </div>
                <div style={{ fontSize:32, fontWeight:800,
                              color: SEVERITY_COLOR[result.severity],
                              textTransform:"uppercase" }}>
                  {result.severity}
                </div>
              </div>

              {/* Defects */}
              <div style={{ background:"#1E293B",
                            borderRadius:12, padding:20,
                            border:"1px solid #334155" }}>
                <div style={{ fontSize:14, fontWeight:600,
                              marginBottom:12,
                              color:"#F1F5F9" }}>
                  Defects Detected
                </div>
                <div style={{ display:"flex",
                              gap:8, flexWrap:"wrap" }}>
                  {result.defects_detected.map(d=>(
                    <span key={d} style={{
                      padding:"6px 14px", borderRadius:20,
                      background: d==="healthy"
                        ?"#1D9E7522":"#EF444422",
                      color: d==="healthy"
                        ?"#1D9E75":"#EF4444",
                      border:`1px solid ${d==="healthy"
                        ?"#1D9E75":"#EF4444"}`,
                      fontSize:13, fontWeight:600,
                    }}>
                      {d}
                    </span>
                  ))}
                </div>
              </div>

              {/* Confidence scores */}
              <div style={{ background:"#1E293B",
                            borderRadius:12, padding:20,
                            border:"1px solid #334155" }}>
                <div style={{ fontSize:14, fontWeight:600,
                              marginBottom:12,
                              color:"#F1F5F9" }}>
                  Confidence Scores
                </div>
                {Object.entries(result.confidence_scores)
                  .map(([k,v])=>(
                  <div key={k} style={{ marginBottom:10 }}>
                    <div style={{ display:"flex",
                                  justifyContent:"space-between",
                                  marginBottom:4 }}>
                      <span style={{ fontSize:13,
                                     color:"#94A3B8",
                                     textTransform:"capitalize" }}>
                        {k.replace("_"," ")}
                      </span>
                      <span style={{ fontSize:13,
                                     fontWeight:600,
                                     color:"#E2E8F0" }}>
                        {(v*100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ background:"#0F172A",
                                  borderRadius:4, height:6 }}>
                      <div style={{ width:`${v*100}%`,
                                    height:"100%",
                                    borderRadius:4,
                                    background: v>0.5
                                      ?"#EF4444":"#38BDF8",
                                    transition:"width 0.5s" }}/>
                    </div>
                  </div>
                ))}
              </div>

              {/* Recommendation */}
              <div style={{ background:"#0F172A",
                            borderRadius:12, padding:16,
                            border:"1px solid #334155" }}>
                <div style={{ fontSize:12, color:"#64748B",
                              marginBottom:6 }}>
                  Recommendation
                </div>
                <div style={{ fontSize:14,
                              color:"#CBD5E1",
                              lineHeight:1.6 }}>
                  {result.recommendation}
                </div>
              </div>

              {/* Annotated image */}
              {result.annotated_image && (
                <div style={{ background:"#1E293B",
                              borderRadius:12, padding:16,
                              border:"1px solid #334155" }}>
                  <div style={{ fontSize:14, fontWeight:600,
                                marginBottom:12,
                                color:"#F1F5F9" }}>
                    Annotated Image
                  </div>
                  <img
                    src={`data:image/jpeg;base64,${result.annotated_image}`}
                    alt="annotated"
                    style={{ width:"100%", borderRadius:8,
                             border:"1px solid #334155" }}/>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
