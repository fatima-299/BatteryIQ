"""
BatteryIQ — BatteryChat API (LLM + RAG)
Endpoint: POST /chat

Uses GPT-4o with RAG:
  1. Retrieve cell data from PostgreSQL
  2. Inject as context into system prompt
  3. Return data-grounded explanation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

router = APIRouter()


class ChatRequest(BaseModel):
    message   : str
    cell_id   : Optional[str] = None
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response  : str
    cell_id   : Optional[str]
    data_used : bool


SYSTEM_PROMPT = """You are BatteryIQ Assistant, an expert battery health analyst.
You analyse lithium-ion battery degradation data and explain findings to fleet managers.

Your responses should be:
- Clear and non-technical when possible
- Based on the actual cell data provided
- Actionable (tell the user what to do)
- Concise (2-3 paragraphs maximum)

Battery health categories:
- Excellent (SOH ≥ 95%): No action needed
- Good (SOH 90-95%): Monitor regularly
- Fair (SOH 80-90%): Schedule inspection
- Poor (SOH 70-80%): Plan replacement soon
- Critical (SOH < 70%): Immediate replacement recommended

Always reference specific numbers from the cell data when available."""


def build_context(cell_id: str) -> str:
    """Retrieve cell data from DB and build context string."""
    try:
        from services.database import get_cell_history, get_cell_latest
        latest  = get_cell_latest(cell_id)
        history = get_cell_history(cell_id)

        if not latest:
            return f"No data found for cell {cell_id}."

        # Build context
        ctx = f"""
CELL DATA FOR: {cell_id}
========================
Current Status:
- SOH: {latest.get('soh_pct', 'N/A')}%
- Risk Score: {latest.get('risk_score', 'N/A')}
- Alert Status: {latest.get('alert_flag', 'N/A')}
- Degradation Category: {latest.get('degradation_category', 'N/A')}
- Source: {latest.get('source', 'N/A')}
- Chemistry: {latest.get('chemistry', 'N/A')}
- Total Cycles: {latest.get('cycle_number', 'N/A')}

Historical Trend:
- Starting SOH: {history['soh_pct'].iloc[0]:.2f}% (cycle 1)
- Current SOH: {history['soh_pct'].iloc[-1]:.2f}% (cycle {history['cycle_number'].iloc[-1]})
- Total SOH Drop: {history['soh_pct'].iloc[0] - history['soh_pct'].iloc[-1]:.2f}%
- Average Fade Rate: {history['capacity_fade_rate'].mean():.4f}% per cycle

Physics Indicators:
- Avg Temperature: {history['avg_temp_c'].mean():.1f}°C
- Internal Resistance: {history['internal_resistance'].mean():.4f} Ω
- Arrhenius Factor: {history['arrhenius_factor'].mean():.6f}
"""
        return ctx
    except Exception as e:
        return f"Could not retrieve data for {cell_id}: {str(e)}"


@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    BatteryChat endpoint — GPT-4o with RAG from PostgreSQL.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Build context
        context   = ""
        data_used = False
        if req.cell_id:
            context   = build_context(req.cell_id)
            data_used = True

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        if context:
            messages.append({
                "role"   : "system",
                "content": f"LIVE BATTERY DATA:\n{context}"
            })

        messages.append({
            "role"   : "user",
            "content": req.message
        })

        # Call GPT-4o
        response = client.chat.completions.create(
            model      = "gpt-4o",
            messages   = messages,
            max_tokens = 500,
            temperature= 0.3,
        )

        answer = response.choices[0].message.content

        return ChatResponse(
            response  = answer,
            cell_id   = req.cell_id,
            data_used = data_used
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cells")
def get_available_cells():
    """Get list of all cell IDs for chat dropdown."""
    try:
        from services.database import query_df
        df = query_df(
            "SELECT DISTINCT cell_id, source, chemistry "
            "FROM battery_cycles ORDER BY cell_id"
        )
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
