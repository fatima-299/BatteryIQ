"""
BatteryIQ — Database Connection
Handles PostgreSQL connection via SQLAlchemy.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/batteryiq"
)

engine        = create_engine(DATABASE_URL)
SessionLocal  = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def query_df(sql: str, params: dict = None) -> pd.DataFrame:
    """Execute SQL and return DataFrame."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def get_cell_history(cell_id: str) -> pd.DataFrame:
    """Get full cycle history for a specific cell."""
    return query_df(
        """
        SELECT cycle_number, soh_pct, cycle_capacity_ah,
               internal_resistance, avg_temp_c, risk_score,
               alert_flag, degradation_category,
               capacity_fade_rate, arrhenius_factor
        FROM battery_cycles
        WHERE cell_id = :cell_id
        ORDER BY cycle_number
        """,
        {"cell_id": cell_id}
    )


def get_fleet_summary() -> pd.DataFrame:
    """Get latest status per cell for fleet overview."""
    return query_df(
        """
        SELECT DISTINCT ON (cell_id)
            cell_id, source, chemistry,
            soh_pct, risk_score, alert_flag,
            degradation_category, cycle_number
        FROM battery_cycles
        ORDER BY cell_id, cycle_number DESC
        """
    )


def get_cell_latest(cell_id: str) -> dict:
    """Get latest cycle data for a cell."""
    df = query_df(
        """
        SELECT *
        FROM battery_cycles
        WHERE cell_id = :cell_id
        ORDER BY cycle_number DESC
        LIMIT 1
        """,
        {"cell_id": cell_id}
    )
    if df.empty:
        return {}
    return df.iloc[0].to_dict()
