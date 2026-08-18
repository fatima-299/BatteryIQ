"""
BatteryIQ — Model Loader
Loads all trained models at startup.
"""

import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = Path(os.getenv("MODEL_DIR", "../../ml/models"))
FEAT_DIR  = Path(os.getenv("FEAT_DIR",  "../../data/features"))

# ── Load models once at startup ────────────────────────────────────────────
print("🔋 Loading BatteryIQ models ...")

# XGBoost v2
xgb_model = joblib.load(MODEL_DIR / "xgboost_v2_soh.pkl")
print("   ✅ XGBoost v2 loaded")

# PINN
from torch import nn

class BatteryPINN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
        )
        self.res1_main = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(256, 256), nn.LayerNorm(256),
        )
        self.res1_act  = nn.GELU()
        self.res2_main = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(128, 128), nn.LayerNorm(128),
        )
        self.res2_proj = nn.Linear(256, 128)
        self.res2_act  = nn.GELU()
        self.head      = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        h  = self.enc(x)
        h  = self.res1_act(self.res1_main(h) + h)
        h2 = self.res2_act(self.res2_main(h) + self.res2_proj(h))
        return self.head(h2).squeeze(-1)

pinn_scaler = joblib.load(MODEL_DIR / "pinn_final_scaler.pkl")
pinn_model  = BatteryPINN(input_dim=19)
state       = torch.load(MODEL_DIR / "pinn_final.pt",
                         map_location="cpu")
pinn_model.load_state_dict(state)
pinn_model.eval()
print("   ✅ PINN loaded")

print("   ✅ All models ready")

# ── Feature columns ────────────────────────────────────────────────────────
XGB_FEATURES = [
    "cycle_number", "cycle_normalized", "lifecycle_stage",
    "cycles_from_start", "cycle_capacity_ah",
    "avg_temp_c", "avg_voltage_v", "avg_current_a",
    "internal_resistance", "capacity_fade_rate",
    "ir_growth_rate", "ir_cumulative_growth",
    "cap_normalized", "arrhenius_factor",
    "src_calce", "src_nasa", "src_stanford",
    "chem_CS2", "chem_CX2", "chem_LFP", "chem_NMC",
]

PINN_FEATURES = [
    "cycle_number", "cycle_normalized", "lifecycle_stage",
    "cycles_from_start", "cycle_capacity_ah",
    "avg_temp_c", "avg_voltage_v", "avg_current_a",
    "internal_resistance", "capacity_fade_rate",
    "ir_growth_rate", "ir_cumulative_growth",
    "cap_normalized", "arrhenius_factor",
    "src_calce", "src_stanford",
    "chem_CS2", "chem_CX2", "chem_LFP",
]


def predict_soh_xgb(features: dict) -> float:
    """Predict SOH using XGBoost v2."""
    df = pd.DataFrame([features])
    for col in XGB_FEATURES:
        if col not in df.columns:
            df[col] = 0
    X      = df[XGB_FEATURES].fillna(0)
    return float(xgb_model.predict(X)[0])


def predict_soh_pinn(features: dict) -> float:
    """Predict SOH using PINN."""
    df = pd.DataFrame([features])
    for col in PINN_FEATURES:
        if col not in df.columns:
            df[col] = 0
    X     = df[PINN_FEATURES].fillna(0).values
    X_sc  = pinn_scaler.transform(X)
    with torch.no_grad():
        pred = pinn_model(torch.FloatTensor(X_sc))
    return float(pred.numpy()[0])
