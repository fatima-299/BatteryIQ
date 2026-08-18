"""
BatteryIQ — Prediction API
Endpoints: POST /predict/soh, POST /predict/rul
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np

router = APIRouter()


class CycleFeatures(BaseModel):
    cell_id           : str
    cycle_number      : float
    cycle_capacity_ah : float
    nominal_capacity_ah: Optional[float] = 1.1
    avg_temp_c        : Optional[float] = 25.0
    avg_voltage_v     : Optional[float] = None
    avg_current_a     : Optional[float] = None
    internal_resistance: Optional[float] = None
    source            : Optional[str] = "stanford"
    chemistry         : Optional[str] = "LFP"
    model             : Optional[str] = "xgboost"  # xgboost or pinn


class SOHResponse(BaseModel):
    cell_id         : str
    cycle_number    : float
    soh_predicted   : float
    model_used      : str
    degradation_category: str
    alert_flag      : str
    risk_score      : float


def get_features(req: CycleFeatures) -> dict:
    """Build feature dict from request."""
    # Encode source
    src_calce    = 1 if req.source == "calce"    else 0
    src_nasa     = 1 if req.source == "nasa"     else 0
    src_stanford = 1 if req.source == "stanford" else 0

    # Encode chemistry
    chem_CS2 = 1 if req.chemistry == "CS2" else 0
    chem_CX2 = 1 if req.chemistry == "CX2" else 0
    chem_LFP = 1 if req.chemistry == "LFP" else 0
    chem_NMC = 1 if req.chemistry == "NMC" else 0

    # Derived features
    cap_normalized   = req.cycle_capacity_ah / (req.nominal_capacity_ah or 1.1)
    cycle_normalized = min(req.cycle_number / 1000, 1.0)
    lifecycle_stage  = (0 if cycle_normalized <= 0.33
                        else 1 if cycle_normalized <= 0.66 else 2)

    # Arrhenius
    import math
    T       = (req.avg_temp_c or 25.0) + 273.15
    arrh    = math.exp(-50000 / (8.314 * T))

    return {
        "cycle_number"        : req.cycle_number,
        "cycle_normalized"    : cycle_normalized,
        "lifecycle_stage"     : lifecycle_stage,
        "cycles_from_start"   : req.cycle_number - 1,
        "cycle_capacity_ah"   : req.cycle_capacity_ah,
        "avg_temp_c"          : req.avg_temp_c or 25.0,
        "avg_voltage_v"       : req.avg_voltage_v or 3.7,
        "avg_current_a"       : req.avg_current_a or -1.5,
        "internal_resistance" : req.internal_resistance or 0.02,
        "capacity_fade_rate"  : 0.0,
        "ir_growth_rate"      : 0.0,
        "ir_cumulative_growth": 0.0,
        "cap_normalized"      : cap_normalized,
        "arrhenius_factor"    : arrh,
        "src_calce"           : src_calce,
        "src_nasa"            : src_nasa,
        "src_stanford"        : src_stanford,
        "chem_CS2"            : chem_CS2,
        "chem_CX2"            : chem_CX2,
        "chem_LFP"            : chem_LFP,
        "chem_NMC"            : chem_NMC,
    }


def categorise(soh: float) -> tuple:
    if soh >= 95:
        cat, flag, risk = "excellent", "OK", (100-soh)*1.0
    elif soh >= 90:
        cat, flag, risk = "good", "OK", (100-soh)*1.5
    elif soh >= 80:
        cat, flag, risk = "fair", "MONITOR", (100-soh)*2.0
    elif soh >= 70:
        cat, flag, risk = "poor", "WARNING", (100-soh)*3.0
    else:
        cat, flag, risk = "critical", "EOL_REACHED", (100-soh)*4.0
    return cat, flag, round(risk, 2)


@router.post("/soh", response_model=SOHResponse)
def predict_soh(req: CycleFeatures):
    """
    Predict SOH% for a battery cycle.
    Use model='xgboost' for fast prediction or model='pinn' for physics-constrained.
    """
    try:
        from services.model_loader import predict_soh_xgb, predict_soh_pinn
        features = get_features(req)

        if req.model == "pinn":
            soh = predict_soh_pinn(features)
            model_used = "PINN (Physics-Informed)"
        else:
            soh = predict_soh_xgb(features)
            model_used = "XGBoost v2"

        soh = float(np.clip(soh, 0, 110))
        cat, flag, risk = categorise(soh)

        return SOHResponse(
            cell_id             = req.cell_id,
            cycle_number        = req.cycle_number,
            soh_predicted       = round(soh, 4),
            model_used          = model_used,
            degradation_category= cat,
            alert_flag          = flag,
            risk_score          = risk,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rul")
def predict_rul(req: CycleFeatures):
    """Estimate RUL based on current SOH and degradation rate."""
    try:
        from services.model_loader import predict_soh_xgb
        features = get_features(req)
        soh      = predict_soh_xgb(features)

        # Estimate RUL: how many cycles until SOH = 80%
        # Using linear extrapolation from current degradation rate
        if soh <= 80:
            rul = 0
        else:
            fade_per_cycle = max(0.001, abs(features["capacity_fade_rate"]))
            rul = int((soh - 80) / (fade_per_cycle * 100 + 0.05))

        return {
            "cell_id"     : req.cell_id,
            "cycle_number": req.cycle_number,
            "soh_current" : round(soh, 2),
            "rul_cycles"  : rul,
            "eol_cycle_estimated": int(req.cycle_number) + rul,
            "confidence"  : "medium"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
