"""
BatteryIQ — Fleet API
Endpoints: GET /fleet, GET /fleet/{cell_id}
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional

router = APIRouter()


@router.get("/")
def get_fleet(source: Optional[str] = None,
              chemistry: Optional[str] = None,
              alert: Optional[str] = None):
    """Get fleet overview — latest status per cell."""
    try:
        from services.database import get_fleet_summary
        df = get_fleet_summary()

        if source:
            df = df[df["source"] == source]
        if chemistry:
            df = df[df["chemistry"] == chemistry]
        if alert:
            df = df[df["alert_flag"] == alert]

        return {
            "total_cells"  : len(df),
            "avg_soh"      : round(df["soh_pct"].mean(), 2),
            "cells_eol"    : int((df["alert_flag"] == "EOL_REACHED").sum()),
            "cells_warning": int((df["alert_flag"] == "WARNING").sum()),
            "cells_monitor": int((df["alert_flag"] == "MONITOR").sum()),
            "cells"        : df.fillna(0).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{cell_id}")
def get_cell(cell_id: str):
    """Get full history and latest status for one cell."""
    try:
        from services.database import get_cell_history, get_cell_latest
        history = get_cell_history(cell_id)
        latest  = get_cell_latest(cell_id)

        if history.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Cell {cell_id} not found"
            )

        return {
            "cell_id"    : cell_id,
            "source"     : latest.get("source"),
            "chemistry"  : latest.get("chemistry"),
            "total_cycles": int(history["cycle_number"].max()),
            "current_soh": round(float(latest.get("soh_pct", 0)), 2),
            "risk_score" : round(float(latest.get("risk_score", 0)), 2),
            "alert_flag" : latest.get("alert_flag"),
            "degradation_category": latest.get("degradation_category"),
            "history"    : history.fillna(0).to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/summary")
def get_stats():
    """Fleet-wide statistics for dashboard."""
    try:
        from services.database import query_df
        stats = query_df("""
            SELECT
                source,
                chemistry,
                COUNT(DISTINCT cell_id) AS n_cells,
                ROUND(AVG(soh_pct)::numeric, 2) AS avg_soh,
                ROUND(AVG(risk_score)::numeric, 2) AS avg_risk,
                SUM(CASE WHEN alert_flag='EOL_REACHED' THEN 1 ELSE 0 END) AS eol_count
            FROM battery_cycles
            GROUP BY source, chemistry
            ORDER BY source
        """)
        return stats.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
