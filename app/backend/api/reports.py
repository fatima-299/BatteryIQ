"""
BatteryIQ — Report Generator API
Endpoint: POST /generate-report/{cell_id}

Generates PDF health report using:
- ReportLab for PDF assembly
- GPT-4o for narrative generation
- Matplotlib for embedded charts
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io
import os

router = APIRouter()


def generate_narrative(cell_data: dict, history_stats: dict) -> str:
    """Generate AI narrative for the report using GPT-4o."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = f"""
Write a professional battery health report narrative for fleet managers.
Be concise (3 short paragraphs), factual, and actionable.

Cell: {cell_data.get('cell_id')}
Chemistry: {cell_data.get('chemistry')} from {cell_data.get('source')}
Current SOH: {cell_data.get('soh_pct')}%
Status: {cell_data.get('degradation_category')} ({cell_data.get('alert_flag')})
Total cycles: {cell_data.get('cycle_number')}
Risk score: {cell_data.get('risk_score')}

Historical stats:
- Starting SOH: {history_stats.get('start_soh', 'N/A')}%
- Total SOH drop: {history_stats.get('soh_drop', 'N/A')}%
- Average fade rate: {history_stats.get('fade_rate', 'N/A')}% per cycle

Write: 1) Current status summary, 2) Degradation analysis, 3) Recommendation
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a battery health report writer for EV fleet managers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception:
        # Fallback narrative if OpenAI unavailable
        soh = cell_data.get('soh_pct', 0)
        cat = cell_data.get('degradation_category', 'unknown')
        return (
            f"Cell {cell_data.get('cell_id')} currently shows a State of Health "
            f"of {soh}%, classified as {cat}. "
            f"The cell has completed {cell_data.get('cycle_number')} cycles "
            f"with a risk score of {cell_data.get('risk_score')}. "
            f"Based on current degradation trends, "
            f"{'immediate replacement is recommended' if soh < 80 else 'continued monitoring is advised'}."
        )


def create_soh_chart(history: list) -> bytes:
    """Generate SOH trajectory chart as PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cycles = [h["cycle_number"] for h in history]
    sohs   = [h["soh_pct"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(cycles, sohs, color="#378ADD", linewidth=1.5)
    ax.axhline(80, color="#EF4444", linestyle="--",
               linewidth=1.5, label="EOL threshold (80%)")
    ax.fill_between(cycles, sohs, 80,
                    where=[s > 80 for s in sohs],
                    alpha=0.1, color="#378ADD")
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("SOH (%)")
    ax.set_title("State of Health Trajectory", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


@router.post("/{cell_id}")
async def generate_report(cell_id: str):
    """
    Generate a PDF health report for a specific battery cell.
    Includes: AI narrative, SOH chart, risk metrics, recommendations.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor, black, white
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, Image as RLImage
        )
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

        # Get cell data
        from services.database import get_cell_history, get_cell_latest
        latest  = get_cell_latest(cell_id)
        history = get_cell_history(cell_id)

        if not latest:
            raise HTTPException(
                status_code=404,
                detail=f"Cell {cell_id} not found"
            )

        history_list  = history.fillna(0).to_dict(orient="records")
        history_stats = {
            "start_soh": round(float(history["soh_pct"].iloc[0]), 2),
            "soh_drop" : round(float(history["soh_pct"].iloc[0] -
                                     history["soh_pct"].iloc[-1]), 2),
            "fade_rate": round(float(history["capacity_fade_rate"].mean()), 4),
        }

        # Generate narrative
        narrative = generate_narrative(latest, history_stats)

        # Generate SOH chart
        chart_bytes = create_soh_chart(history_list[-200:])  # last 200 cycles

        # Build PDF
        buffer = io.BytesIO()
        doc    = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )
        styles  = getSampleStyleSheet()
        story   = []

        # ── Colors ────────────────────────────────────────────────────
        BLUE    = HexColor("#378ADD")
        GREEN   = HexColor("#1D9E75")
        RED     = HexColor("#EF4444")
        ORANGE  = HexColor("#EF9F27")
        DARK    = HexColor("#2D3748")

        status_color = {
            "excellent": GREEN,
            "good"     : GREEN,
            "fair"     : ORANGE,
            "poor"     : ORANGE,
            "critical" : RED,
        }.get(latest.get("degradation_category",""), BLUE)

        # ── Header ────────────────────────────────────────────────────
        header_style = ParagraphStyle(
            "header",
            fontSize=22, textColor=white,
            alignment=TA_CENTER, fontName="Helvetica-Bold",
            spaceAfter=6
        )
        sub_style = ParagraphStyle(
            "sub",
            fontSize=11, textColor=white,
            alignment=TA_CENTER, fontName="Helvetica",
        )
        header_data = [[
            Paragraph("🔋 BatteryIQ Health Report", header_style),
        ]]
        header_table = Table(header_data, colWidths=[17*cm])
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), DARK),
            ("TOPPADDING",    (0,0), (-1,-1), 15),
            ("BOTTOMPADDING", (0,0), (-1,-1), 15),
            ("ROUNDEDCORNERS", [5]),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 0.4*cm))

        # Subtitle
        story.append(Paragraph(
            f"Cell: <b>{cell_id}</b> | "
            f"Chemistry: <b>{latest.get('chemistry','N/A')}</b> | "
            f"Source: <b>{latest.get('source','N/A')}</b>",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.3*cm))

        # ── KPI Table ─────────────────────────────────────────────────
        kpi_data = [
            ["Metric", "Value", "Status"],
            ["State of Health (SOH)",
             f"{latest.get('soh_pct','N/A')}%",
             latest.get("degradation_category","N/A").upper()],
            ["Risk Score",
             str(latest.get("risk_score","N/A")),
             latest.get("alert_flag","N/A")],
            ["Total Cycles",
             str(int(latest.get("cycle_number", 0))),
             "completed"],
            ["Starting SOH",
             f"{history_stats['start_soh']}%",
             "baseline"],
            ["SOH Drop",
             f"{history_stats['soh_drop']}%",
             "total degradation"],
        ]
        kpi_table = Table(kpi_data,
                          colWidths=[7*cm, 5*cm, 5*cm])
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), DARK),
            ("TEXTCOLOR",     (0,0), (-1,0), white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [HexColor("#F7FAFC"), white]),
            ("GRID",          (0,0), (-1,-1), 0.5, HexColor("#E2E8F0")),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("BACKGROUND",    (2,1), (2,1), status_color),
            ("TEXTCOLOR",     (2,1), (2,1), white),
            ("FONTNAME",      (2,1), (2,1), "Helvetica-Bold"),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 0.4*cm))

        # ── SOH Chart ─────────────────────────────────────────────────
        chart_img = io.BytesIO(chart_bytes)
        story.append(RLImage(chart_img, width=15*cm, height=6*cm))
        story.append(Spacer(1, 0.3*cm))

        # ── AI Narrative ──────────────────────────────────────────────
        story.append(Paragraph(
            "AI-Generated Health Assessment",
            ParagraphStyle("section", fontSize=13,
                           fontName="Helvetica-Bold",
                           textColor=BLUE, spaceAfter=8)
        ))
        for para in narrative.split("\n\n"):
            if para.strip():
                story.append(Paragraph(
                    para.strip(),
                    ParagraphStyle("body", fontSize=10,
                                   leading=14, alignment=TA_JUSTIFY)
                ))
                story.append(Spacer(1, 0.2*cm))

        # ── Footer ────────────────────────────────────────────────────
        story.append(Spacer(1, 0.5*cm))
        footer_data = [[
            Paragraph(
                "Generated by BatteryIQ | "
                "Physics-Informed ML for EV Battery Degradation | "
                "Confidential",
                ParagraphStyle("footer", fontSize=8,
                               textColor=HexColor("#718096"),
                               alignment=TA_CENTER)
            )
        ]]
        footer_table = Table(footer_data, colWidths=[17*cm])
        footer_table.setStyle(TableStyle([
            ("TOPBORDER",  (0,0), (-1,0), 0.5, HexColor("#E2E8F0")),
            ("TOPPADDING", (0,0), (-1,0), 8),
        ]))
        story.append(footer_table)

        # Build PDF
        doc.build(story)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    f"attachment; filename=BatteryIQ_{cell_id}_report.pdf"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
