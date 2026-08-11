"""
BatteryIQ — Step 15: HNEI Independent Validation (v2)
======================================================
Validates XGBoost v2 on HNEI dataset (Hawaii Natural Energy Institute)
- 14 NMC-LCO cells, ~1,076 cycles each
- Completely independent from training data
- Different institution, different lab, different protocol

SOH computation fix:
  RUL=max_RUL → SOH=100% (brand new battery)
  RUL=0       → SOH=80%  (end of life threshold)
  SOH = 80 + (RUL / max_RUL) × 20

Run from BatteryIQ root:
  python ml/validation/11_hnei_validation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).resolve().parents[2]
HNEI_DIR = ROOT / "data" / "raw" / "hnei"
MOD_DIR  = ROOT / "ml" / "models"
EVAL_DIR = ROOT / "ml" / "evaluation"
FIG_DIR  = ROOT / "memoire" / "figures"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "cycle_number", "cycle_normalized", "lifecycle_stage",
    "cycles_from_start", "cycle_capacity_ah",
    "avg_temp_c", "avg_voltage_v", "avg_current_a",
    "internal_resistance", "capacity_fade_rate",
    "ir_growth_rate", "ir_cumulative_growth",
    "cap_normalized", "arrhenius_factor",
    "src_calce", "src_nasa", "src_stanford",
    "chem_CS2", "chem_CX2", "chem_LFP", "chem_NMC",
]


# ── Load HNEI ──────────────────────────────────────────────────────────────
def load_hnei():
    print("📂 Loading HNEI dataset ...")
    files = sorted(HNEI_DIR.glob("HNEI_*_features.csv"))
    if not files:
        raise FileNotFoundError(f"No HNEI files in {HNEI_DIR}")

    dfs = []
    for f in files:
        df           = pd.read_csv(f, index_col=0)
        df["cell_id"] = f"hnei_{f.stem.replace('_features','')}"
        dfs.append(df)

    hnei = pd.concat(dfs, ignore_index=True)
    print(f"   Cells  : {hnei['cell_id'].nunique()}")
    print(f"   Cycles : {len(hnei):,}")
    return hnei


# ── Engineer features ──────────────────────────────────────────────────────
def engineer_features(hnei):
    """
    Map HNEI columns to BatteryIQ feature schema.

    SOH fix:
      HNEI cells cycle until 80% capacity (EOL threshold).
      RUL = cycles remaining before 80% SOH.
      Therefore:
        RUL = max_RUL → SOH = 100% (fresh battery)
        RUL = 0       → SOH = 80%  (end of life)
        SOH = 80 + (RUL / max_RUL) × 20

    Capacity fix:
      HNEI nominal capacity = 2.8 Ah
      cycle_capacity_ah = (Discharge Time / first_cycle_discharge_time) × 2.8
      This correctly normalises capacity relative to first cycle.
    """
    print("\n⚙️  Engineering features ...")

    NOMINAL_CAP = 2.8   # Ah — HNEI LG ICR18650 C2 cells
    R           = 8.314
    Ea          = 50000.0
    T_kelvin    = 25.0 + 273.15

    all_cells = []
    for cell_id, grp in hnei.groupby("cell_id"):
        grp = grp.sort_values("Cycle_Index").reset_index(drop=True)
        n   = len(grp)

        # ── SOH from RUL (corrected) ───────────────────────────────────
        max_rul         = grp["RUL"].max()
        grp["soh_pct"]  = 80.0 + (grp["RUL"] / max_rul) * 20.0
        # Clip to valid range
        grp["soh_pct"]  = grp["soh_pct"].clip(80.0, 100.0)

        # ── Cycle position features ────────────────────────────────────
        grp["cycle_number"]     = grp["Cycle_Index"].values.astype(float)
        grp["cycles_from_start"] = grp["cycle_number"] - 1
        grp["cycle_normalized"] = (grp["cycle_number"] - 1) / (n - 1 + 1e-8)
        norm                    = grp["cycle_normalized"].values
        grp["lifecycle_stage"]  = np.where(
            norm <= 0.33, 0, np.where(norm <= 0.66, 1, 2)
        )

        # ── Capacity from Discharge Time ───────────────────────────────
        first_disch = grp["Discharge Time (s)"].iloc[0]
        grp["cycle_capacity_ah"] = (
            grp["Discharge Time (s)"] / first_disch
        ) * NOMINAL_CAP

        # Cap normalized
        grp["cap_normalized"] = grp["cycle_capacity_ah"] / NOMINAL_CAP

        # ── Capacity fade rate (rolling slope over 10 cycles) ──────────
        cap        = grp["cycle_capacity_ah"].values
        fade_rates = np.zeros(n)
        for i in range(3, n):
            start = max(0, i - 10)
            y     = cap[start:i]
            x     = np.arange(len(y))
            if len(y) >= 3:
                try:
                    fade_rates[i] = np.polyfit(x, y, 1)[0]
                except Exception:
                    fade_rates[i] = 0.0
        grp["capacity_fade_rate"] = fade_rates

        # ── Voltage features ───────────────────────────────────────────
        grp["avg_voltage_v"] = grp["Max. Voltage Dischar. (V)"].values

        # ── Temperature — HNEI tested at constant 25°C ────────────────
        grp["avg_temp_c"] = 25.0

        # ── Arrhenius factor (constant at 25°C) ───────────────────────
        grp["arrhenius_factor"] = np.exp(-Ea / (R * T_kelvin))

        # ── Physics features not available in HNEI ────────────────────
        # Internal resistance → impute with 0.02 (typical NMC value)
        grp["internal_resistance"]  = 0.02
        grp["ir_growth_rate"]       = 0.0
        grp["ir_cumulative_growth"] = 0.0

        # ── Current not available ──────────────────────────────────────
        grp["avg_current_a"] = -2.8   # 1C discharge = 2.8A (nominal)

        # ── Source/chemistry encoding ──────────────────────────────────
        # NMC-LCO blend → encode as NMC (closest chemistry)
        grp["src_calce"]    = 0
        grp["src_nasa"]     = 0
        grp["src_stanford"] = 0
        grp["chem_CS2"]     = 0
        grp["chem_CX2"]     = 0
        grp["chem_LFP"]     = 0
        grp["chem_NMC"]     = 1

        all_cells.append(grp)

    df = pd.concat(all_cells, ignore_index=True)
    print(f"   SOH range : {df['soh_pct'].min():.1f}% → "
          f"{df['soh_pct'].max():.1f}%")
    print(f"   Cap range : {df['cycle_capacity_ah'].min():.3f} → "
          f"{df['cycle_capacity_ah'].max():.3f} Ah")
    print(f"   Cells     : {df['cell_id'].nunique()}")
    return df


# ── Validate ───────────────────────────────────────────────────────────────
def validate(df):
    print("\n🔍 Validating XGBoost v2 on HNEI ...")
    model = joblib.load(MOD_DIR / "xgboost_v2_soh.pkl")

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["soh_pct"].values

    y_pred = model.predict(X)
    # Clip predictions to valid SOH range
    y_pred = np.clip(y_pred, 75.0, 105.0)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae  = mean_absolute_error(y, y_pred)
    r2   = r2_score(y, y_pred)
    # Safe MAPE — avoid near-zero SOH values
    mask = y > 5.0
    mape = np.mean(np.abs((y[mask]-y_pred[mask]) / y[mask])) * 100

    print(f"   RMSE : {rmse:.4f}%")
    print(f"   MAE  : {mae:.4f}%")
    print(f"   R²   : {r2:.4f}")
    print(f"   MAPE : {mape:.4f}%")

    return y_pred, {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_results(df, y_pred, metrics):
    print("\n📈 Generating validation figure ...")
    df = df.copy()
    df["soh_pred"] = y_pred

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. SOH trajectories
    ax1    = fig.add_subplot(gs[0, :2])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (cell_id, grp) in enumerate(df.groupby("cell_id")):
        grp   = grp.sort_values("cycle_number")
        color = colors[i % 10]
        ax1.plot(grp["cycle_number"], grp["soh_pct"],
                 color=color, linewidth=1.2, alpha=0.7)
        ax1.plot(grp["cycle_number"], grp["soh_pred"],
                 color=color, linewidth=1.2, alpha=0.5,
                 linestyle="--")
    ax1.axhline(80, color="#EF4444", linestyle="--",
                linewidth=2, label="EOL threshold (80%)")
    ax1.set_xlabel("Cycle Number")
    ax1.set_ylabel("SOH (%)")
    ax1.set_title(
        f"HNEI Validation — XGBoost v2\n"
        f"Solid=Actual | Dashed=Predicted | "
        f"RMSE={metrics['rmse']:.4f}% | R²={metrics['r2']:.4f}",
        fontweight="bold"
    )
    ax1.legend(fontsize=8)
    ax1.set_ylim(75, 105)

    # 2. Predicted vs Actual
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(df["soh_pct"], df["soh_pred"],
                alpha=0.3, s=8, color="#1D9E75")
    lims = [79, 101]
    ax2.plot(lims, lims, "r--", linewidth=2, label="Perfect")
    ax2.set_xlabel("Actual SOH (%)")
    ax2.set_ylabel("Predicted SOH (%)")
    ax2.set_title(f"Predicted vs Actual\nR²={metrics['r2']:.4f}",
                  fontweight="bold")
    ax2.legend(fontsize=8)

    # 3. Residuals
    ax3 = fig.add_subplot(gs[1, 0])
    res = df["soh_pred"] - df["soh_pct"]
    ax3.hist(res, bins=50, color="#1D9E75",
             alpha=0.8, edgecolor="white")
    ax3.axvline(0, color="#EF4444", linestyle="--", linewidth=2)
    ax3.axvline(res.mean(), color="#EF9F27", linestyle="--",
                linewidth=1.5, label=f"Mean={res.mean():.3f}%")
    ax3.set_xlabel("Residual (%)")
    ax3.set_ylabel("Count")
    ax3.set_title("Residual Distribution", fontweight="bold")
    ax3.legend(fontsize=8)

    # 4. Comparison table
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis("off")
    table_data = [
        ["XGBoost v2", "Stanford LFP",
         "0.1055%", "0.9994", "Training chemistry"],
        ["XGBoost v2", "CALCE LiCoO2",
         "0.5656%", "0.9991", "Training chemistry"],
        ["XGBoost v2", "NASA NMC",
         "2.1634%", "0.9898", "Held-out — diff. lab"],
        ["XGBoost v2", "HNEI NMC-LCO ★",
         f"{metrics['rmse']:.4f}%",
         f"{metrics['r2']:.4f}",
         "Independent validation ★"],
    ]
    tbl = ax4.table(
        cellText  = table_data,
        colLabels = ["Model", "Dataset", "RMSE", "R²", "Status"],
        cellLoc   = "center", loc="center", bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2D3748")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 4:
            cell.set_facecolor("#C6F6D5")
        elif r % 2 == 0:
            cell.set_facecolor("#F7FAFC")
    ax4.set_title(
        "BatteryIQ — Cross-Dataset Generalisation\n"
        "★ = Independent validation (never seen during training)",
        fontweight="bold", fontsize=11, pad=10
    )

    plt.suptitle(
        "BatteryIQ — HNEI Independent Validation\n"
        "14 NMC-LCO cells | Hawaii Natural Energy Institute | "
        "Zero overlap with training data",
        fontsize=13, fontweight="bold"
    )
    plt.savefig(FIG_DIR / "fig37_hnei_validation.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved → fig37_hnei_validation.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — HNEI Independent Validation v2")
    print("="*60)
    print("   SOH formula : 80 + (RUL/max_RUL) × 20")
    print("   Capacity    : discharge_time / first_cycle × 2.8 Ah")
    print("   IR imputed  : 0.02 Ω (typical NMC value)")
    print("   Temperature : 25°C constant (HNEI protocol)")

    # Load
    hnei = load_hnei()

    # Engineer features
    df = engineer_features(hnei)

    # Validate
    y_pred, metrics = validate(df)

    # Plot
    plot_results(df, y_pred, metrics)

    # Save
    pd.DataFrame([{
        "model"   : "XGBoost_v2",
        "dataset" : "HNEI_NMC_LCO",
        "n_cells" : df["cell_id"].nunique(),
        "n_cycles": len(df),
        "rmse"    : metrics["rmse"],
        "mae"     : metrics["mae"],
        "r2"      : metrics["r2"],
        "mape"    : metrics["mape"],
        "note"    : "Independent validation — never in training"
    }]).to_csv(EVAL_DIR / "hnei_validation.csv", index=False)

    print("\n" + "="*60)
    print("✅ HNEI VALIDATION COMPLETE")
    print("="*60)
    print(f"\n   CROSS-DATASET GENERALISATION SUMMARY:")
    print(f"   {'Dataset':<28} {'RMSE':>8} {'R²':>8}")
    print(f"   {'-'*48}")
    print(f"   {'Stanford LFP (training)':<28} {'0.1055%':>8} {'0.9994':>8}")
    print(f"   {'CALCE LiCoO2 (training)':<28} {'0.5656%':>8} {'0.9991':>8}")
    print(f"   {'NASA NMC (held-out)':<28} {'2.1634%':>8} {'0.9898':>8}")
    print(f"   {'HNEI NMC-LCO (★ indep.)':<28} "
          f"{metrics['rmse']:>8.4f}% {metrics['r2']:>8.4f}")

    print(f"\n   Scientific interpretation:")
    if metrics["rmse"] < 3.0:
        print(f"   🎉 Excellent cross-dataset generalisation!")
        print(f"   XGBoost v2 transfers to unseen NMC-LCO chemistry.")
    elif metrics["rmse"] < 6.0:
        print(f"   ✅ Good cross-dataset generalisation.")
        print(f"   XGBoost v2 transfers reasonably to HNEI NMC-LCO.")
        print(f"   Higher error than training data expected —")
        print(f"   IR and temperature features missing in HNEI.")
    else:
        print(f"   📊 Cross-dataset transfer is limited.")
        print(f"   Missing physics features (IR, temperature variation)")
        print(f"   reduce generalisation to unseen chemistry.")
        print(f"   This is documented as future work in Chapter 7.")

    print(f"\n   Figure → memoire/figures/fig37_hnei_validation.png")
    print(f"   Results → ml/evaluation/hnei_validation.csv")


if __name__ == "__main__":
    main()
