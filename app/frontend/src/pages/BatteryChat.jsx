import { useState, useEffect, useRef } from "react";

const API = "http://localhost:8000";

export default function BatteryChat() {
  const [messages, setMessages] = useState([{
    role:"assistant",
    text:"👋 Hello! I'm BatteryIQ Assistant. Select a battery cell and ask me anything about its health, degradation, or maintenance needs."
  }]);
  const [input,    setInput]    = useState("");
  const [cellId,   setCellId]   = useState("");
  const [cells,    setCells]    = useState([]);
  const [loading,  setLoading]  = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/chat/cells`)
      .then(r=>r.json())
      .then(d=>setCells(d))
      .catch(()=>{});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior:"smooth" });
  }, [messages]);

  const send = async () => {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput("");
    setMessages(prev=>[...prev,{role:"user",text:userMsg}]);
    setLoading(true);

    try {
      const res = await fetch(`${API}/chat/`, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          message: userMsg,
          cell_id: cellId || undefined,
          session_id: "web"
        })
      });
      const data = await res.json();
      setMessages(prev=>[...prev,{
        role:"assistant",
        text: data.response || "Sorry, I could not process that.",
        dataUsed: data.data_used
      }]);
    } catch {
      setMessages(prev=>[...prev,{
        role:"assistant",
        text:"❌ Cannot connect to BatteryChat. Make sure the API is running and your OpenAI key is configured."
      }]);
    }
    setLoading(false);
  };

  const SUGGESTIONS = [
    "What is the current health of this cell?",
    "Should I replace this battery?",
    "What is causing the degradation?",
    "How many cycles remain before EOL?",
    "Compare this cell to fleet average",
  ];

  return (
    <div style={{ display:"flex", flexDirection:"column",
                  height:"100vh", padding:32,
                  color:"#E2E8F0", boxSizing:"border-box" }}>

      {/* Header */}
      <div style={{ marginBottom:20 }}>
        <h1 style={{ fontSize:26, fontWeight:800,
                     color:"#F1F5F9", margin:0 }}>
          💬 BatteryChat
        </h1>
        <p style={{ color:"#64748B", fontSize:14,
                    marginTop:6 }}>
          AI-powered battery health assistant with real-time data context
        </p>
      </div>

      {/* Cell selector */}
      <div style={{ marginBottom:16 }}>
        <label style={{ fontSize:13, color:"#94A3B8",
                        marginBottom:6, display:"block" }}>
          Select cell for context (optional):
        </label>
        <select
          value={cellId}
          onChange={e=>setCellId(e.target.value)}
          style={{ padding:"10px 14px", borderRadius:8,
                   border:"1px solid #334155",
                   background:"#1E293B", color:"#E2E8F0",
                   fontSize:14, width:320, cursor:"pointer" }}>
          <option value="">No cell selected (general questions)</option>
          {cells.map(c=>(
            <option key={c.cell_id} value={c.cell_id}>
              {c.cell_id} — {c.chemistry} ({c.source})
            </option>
          ))}
        </select>
        {cellId && (
          <span style={{ marginLeft:12, fontSize:12,
                         color:"#1D9E75" }}>
            ✅ Cell data will be injected into AI context
          </span>
        )}
      </div>

      {/* Messages */}
      <div style={{ flex:1, overflowY:"auto",
                    background:"#1E293B", borderRadius:12,
                    padding:20, marginBottom:16,
                    border:"1px solid #334155" }}>
        {messages.map((m,i)=>(
          <div key={i}
            style={{ marginBottom:16,
                     display:"flex",
                     justifyContent: m.role==="user"
                       ?"flex-end":"flex-start" }}>
            <div style={{
              maxWidth:"75%",
              padding:"12px 16px", borderRadius:12,
              background: m.role==="user"?"#0F4C81":"#0F172A",
              border:`1px solid ${m.role==="user"
                ?"#38BDF8":"#334155"}`,
              fontSize:14, lineHeight:1.6,
              color: m.role==="user"?"#E2E8F0":"#CBD5E1",
              whiteSpace:"pre-wrap",
            }}>
              {m.role==="assistant" && (
                <div style={{ fontSize:11, color:"#475569",
                              marginBottom:6 }}>
                  🤖 BatteryIQ Assistant
                  {m.dataUsed &&
                    <span style={{ marginLeft:8, color:"#1D9E75" }}>
                      • using live cell data
                    </span>}
                </div>
              )}
              {m.text}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ display:"flex", gap:6,
                        padding:"12px 16px",
                        background:"#0F172A",
                        borderRadius:12, width:80,
                        border:"1px solid #334155" }}>
            {[0,1,2].map(i=>(
              <div key={i}
                style={{ width:8, height:8,
                         borderRadius:"50%",
                         background:"#38BDF8",
                         animation:`bounce 1s ${i*0.2}s infinite` }}/>
            ))}
          </div>
        )}
        <div ref={bottomRef}/>
      </div>

      {/* Suggestions */}
      <div style={{ display:"flex", gap:8,
                    flexWrap:"wrap", marginBottom:12 }}>
        {SUGGESTIONS.map(s=>(
          <button key={s}
            onClick={()=>setInput(s)}
            style={{ padding:"6px 12px", borderRadius:20,
                     background:"#0F172A",
                     border:"1px solid #334155",
                     color:"#64748B", cursor:"pointer",
                     fontSize:12, transition:"all 0.2s" }}
            onMouseEnter={e=>{
              e.target.style.borderColor="#38BDF8";
              e.target.style.color="#38BDF8";}}
            onMouseLeave={e=>{
              e.target.style.borderColor="#334155";
              e.target.style.color="#64748B";}}>
            {s}
          </button>
        ))}
      </div>

      {/* Input */}
      <div style={{ display:"flex", gap:12 }}>
        <input
          value={input}
          onChange={e=>setInput(e.target.value)}
          onKeyDown={e=>e.key==="Enter"&&send()}
          placeholder={cellId
            ? `Ask about ${cellId}...`
            : "Ask about battery health, degradation, maintenance..."}
          style={{ flex:1, padding:"14px 16px",
                   borderRadius:8,
                   border:"1px solid #334155",
                   background:"#1E293B", color:"#E2E8F0",
                   fontSize:14, outline:"none" }}
        />
        <button onClick={send} disabled={loading}
          style={{ padding:"14px 28px", borderRadius:8,
                   background: loading?"#1E293B":"#0F4C81",
                   border:"1px solid #38BDF8",
                   color:"#38BDF8", cursor: loading
                     ?"not-allowed":"pointer",
                   fontSize:14, fontWeight:600,
                   opacity: loading?0.5:1 }}>
          {loading?"...":"Send →"}
        </button>
      </div>

      <style>{`
        @keyframes bounce {
          0%,100%{transform:translateY(0)}
          50%{transform:translateY(-6px)}
        }
      `}</style>
    </div>
  );
}
