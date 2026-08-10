"""
BatteryIQ — SHAP Analysis: Publication-Quality Feature Importance
=================================================================
Produces 5 figures for Chapter 5:
  fig32_shap_summary.png          — overall feature importance bar
  fig33_shap_beeswarm.png         — beeswarm plot (direction of impact)
  fig34_shap_dependence.png       — top 4 feature dependence plots
  fig35_shap_per_chemistry.png    — SHAP importance per chemistry
  fig36_shap_waterfall_sample.png — single cell prediction explained

Run from BatteryIQ root:
  python ml/explainability/10_shap_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

import shap

ROOT     = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data" / "features"
MOD_DIR  = ROOT / "ml" / "models"
EVAL_DIR = ROOT / "ml" / "evaluation"
FIG_DIR  = ROOT / "memoire" / "figures"
EXPL_DIR = ROOT / "ml" / "explainability"
EXPL_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature names (same as XGBoost v2) ────────────────────────────────────
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

# Human-readable feature labels for figures
FEATURE_LABELS = {
    "cycle_number"         : "Cycle Number",
    "cycle_normalized"     : "Normalised Cycle Position",
    "lifecycle_stage"      : "Lifecycle Stage",
    "cycles_from_start"    : "Cycles from Start",
    "cycle_capacity_ah"    : "Discharge Capacity (Ah)",
    "avg_temp_c"           : "Average Temperature (°C)",
    "avg_voltage_v"        : "Average Voltage (V)",
    "avg_current_a"        : "Average Current (A)",
    "internal_resistance"  : "Internal Resistance (Ω)",
    "capacity_fade_rate"   : "Capacity Fade Rate",
    "ir_growth_rate"       : "IR Growth Rate",
    "ir_cumulative_growth" : "Cumulative IR Growth",
    "cap_normalized"       : "Normalised Capacity",
    "arrhenius_factor"     : "Arrhenius Factor (exp(-Ea/RT))",
    "src_calce"            : "Source: CALCE",
    "src_nasa"             : "Source: NASA",
    "src_stanford"         : "Source: Stanford",
    "chem_CS2"             : "Chemistry: CS2 (LiCoO2)",
    "chem_CX2"             : "Chemistry: CX2 (LiCoO2)",
    "chem_LFP"             : "Chemistry: LFP",
    "chem_NMC"             : "Chemistry: NMC",
}

COLORS = {
    "nasa"    : "#378ADD",
    "stanford": "#1D9E75",
    "calce"   : "#EF9F27",
    "physics" : "#7F77DD",
    "temporal": "#EF4444",
    "source"  : "#8B5CF6",
}


def load_data_and_model():
    print("📂 Loading data and model ...")
    df    = pd.read_csv(
        FEAT_DIR / "spark_output" / "feature_matrix_enriched.csv"
    )
    model = joblib.load(MOD_DIR / "xgboost_v2_soh.pkl")

    feature_cols = [c for c in FEATURES if c in df.columns]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["soh_pct"]

    print(f"   Rows     : {len(df):,}")
    print(f"   Features : {len(feature_cols)}")
    print(f"   Model    : XGBoost v2 (no lags — realistic scenario)")
    return df, model, X, y, feature_cols


def compute_shap(model, X, feature_cols, n_sample=3000):
    print(f"\n🔍 Computing SHAP values (sample={n_sample:,}) ...")
    sample = X.sample(n_sample, random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    expected    = explainer.expected_value
    print(f"   SHAP matrix shape: {shap_values.shape}")
    print(f"   Expected value (base): {expected:.2f}%")
    return shap_values, sample, expected


# ── Figure 1: Summary Bar (Mean |SHAP|) ───────────────────────────────────
def plot_shap_summary_bar(shap_values, feature_cols):
    print("\n📊 Figure 1: SHAP Summary Bar ...")

    mean_shap = np.abs(shap_values).mean(axis=0)
    labels    = [FEATURE_LABELS.get(f, f) for f in feature_cols]
    imp_df    = pd.DataFrame({
        "feature"   : feature_cols,
        "label"     : labels,
        "mean_shap" : mean_shap
    }).sort_values("mean_shap", ascending=True)

    # Colour by feature group
    def get_color(feat):
        if feat in ["cycle_number","cycle_normalized","lifecycle_stage",
                    "cycles_from_start"]:
            return COLORS["temporal"]
        elif feat in ["capacity_fade_rate","ir_growth_rate",
                      "ir_cumulative_growth","cap_normalized",
                      "arrhenius_factor","internal_resistance"]:
            return COLORS["physics"]
        elif feat.startswith("src_"):
            return COLORS["source"]
        elif feat.startswith("chem_"):
            return "#F97316"
        else:
            return COLORS["nasa"]

    colors = [get_color(f) for f in imp_df["feature"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars    = ax.barh(imp_df["label"], imp_df["mean_shap"],
                      color=colors, alpha=0.85, edgecolor="white")

    # Value labels
    for bar, val in zip(bars, imp_df["mean_shap"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_xlabel("Mean |SHAP Value| — Average Impact on SOH Prediction (%)")
    ax.set_title("BatteryIQ — Feature Importance (SHAP)\nXGBoost v2 on 134,938 cycles",
                 fontweight="bold", fontsize=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["temporal"], label="Temporal features"),
        Patch(facecolor=COLORS["nasa"],     label="Raw measurements"),
        Patch(facecolor=COLORS["physics"],  label="Physics-derived features"),
        Patch(facecolor=COLORS["source"],   label="Source encoding"),
        Patch(facecolor="#F97316",          label="Chemistry encoding"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig32_shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → fig32_shap_summary.png")

    # Print top 10
    print("\n   Top 10 most important features:")
    for _, row in imp_df.tail(10).iloc[::-1].iterrows():
        bar = "█" * int(row["mean_shap"] * 30)
        print(f"   {row['label']:35s}: {row['mean_shap']:.4f}  {bar}")

    return imp_df


# ── Figure 2: Beeswarm Plot ────────────────────────────────────────────────
def plot_beeswarm(shap_values, X_sample, feature_cols):
    print("\n📊 Figure 2: SHAP Beeswarm ...")

    labels = [FEATURE_LABELS.get(f, f) for f in feature_cols]
    mean_shap = np.abs(shap_values).mean(axis=0)
    order     = np.argsort(mean_shap)[-15:]  # top 15

    fig, ax = plt.subplots(figsize=(12, 8))

    for plot_idx, feat_idx in enumerate(order):
        shap_vals  = shap_values[:, feat_idx]
        feat_vals  = X_sample.iloc[:, feat_idx].values

        # Normalise feature values to [0,1] for coloring
        feat_min  = feat_vals.min()
        feat_max  = feat_vals.max()
        if feat_max > feat_min:
            feat_norm = (feat_vals - feat_min) / (feat_max - feat_min)
        else:
            feat_norm = np.zeros_like(feat_vals)

        # Add jitter for beeswarm effect
        jitter = np.random.uniform(-0.3, 0.3, len(shap_vals))
        y_pos  = plot_idx + jitter

        scatter = ax.scatter(
            shap_vals, y_pos,
            c=feat_norm, cmap="RdBu_r",
            alpha=0.4, s=8,
            vmin=0, vmax=1
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [FEATURE_LABELS.get(feature_cols[i], feature_cols[i])
         for i in order],
        fontsize=9
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("SHAP Value (impact on SOH prediction %)")
    ax.set_title(
        "BatteryIQ — SHAP Beeswarm Plot\n"
        "Red = high feature value, Blue = low feature value",
        fontweight="bold", fontsize=12
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Feature value\n(normalised)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Medium", "High"])

    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig33_shap_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → fig33_shap_beeswarm.png")


# ── Figure 3: Dependence Plots ─────────────────────────────────────────────
def plot_dependence(shap_values, X_sample, feature_cols, imp_df):
    print("\n📊 Figure 3: SHAP Dependence Plots ...")

    # Pick top 4 most important features
    top4 = list(imp_df.tail(4)["feature"].values)[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, feat in zip(axes.flat, top4):
        if feat not in feature_cols:
            continue
        feat_idx  = feature_cols.index(feat)
        shap_vals = shap_values[:, feat_idx]
        feat_vals = X_sample[feat].values

        # Colour by cycle_number if available
        color_feat = "cycle_number"
        if color_feat in feature_cols:
            color_idx  = feature_cols.index(color_feat)
            color_vals = X_sample[color_feat].values
            sc = ax.scatter(feat_vals, shap_vals,
                            c=color_vals, cmap="viridis",
                            alpha=0.4, s=8)
            plt.colorbar(sc, ax=ax, label="Cycle number", shrink=0.8)
        else:
            ax.scatter(feat_vals, shap_vals,
                       color="#378ADD", alpha=0.4, s=8)

        # Trend line
        z = np.polyfit(feat_vals, shap_vals, 2)
        p = np.poly1d(z)
        x_line = np.linspace(feat_vals.min(), feat_vals.max(), 100)
        ax.plot(x_line, p(x_line), color="#EF4444",
                linewidth=2, label="Trend")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(FEATURE_LABELS.get(feat, feat))
        ax.set_ylabel("SHAP Value")
        ax.set_title(f"SHAP Dependence: {FEATURE_LABELS.get(feat, feat)}",
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "BatteryIQ — SHAP Dependence Plots\n"
        "How each feature's value affects its impact on SOH prediction",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = FIG_DIR / "fig34_shap_dependence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → fig34_shap_dependence.png")


# ── Figure 4: SHAP per Chemistry ──────────────────────────────────────────
def plot_shap_per_chemistry(model, df, feature_cols):
    print("\n📊 Figure 4: SHAP per Chemistry ...")

    chemistries = {
        "NMC (NASA)"      : df[df["source"] == "nasa"],
        "LFP (Stanford)"  : df[df["source"] == "stanford"],
        "LiCoO2 (CALCE)"  : df[df["source"] == "calce"],
    }
    chem_colors = ["#378ADD", "#1D9E75", "#EF9F27"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    explainer = shap.TreeExplainer(model)

    all_mean_shaps = {}
    for (chem_name, chem_df), color, ax in zip(
        chemistries.items(), chem_colors, axes
    ):
        sample    = chem_df[feature_cols].fillna(
            chem_df[feature_cols].median()
        ).sample(min(500, len(chem_df)), random_state=42)

        sv        = explainer.shap_values(sample)
        mean_shap = np.abs(sv).mean(axis=0)
        labels    = [FEATURE_LABELS.get(f, f) for f in feature_cols]

        imp       = pd.DataFrame({
            "label": labels,
            "mean_shap": mean_shap
        }).sort_values("mean_shap", ascending=True).tail(10)

        all_mean_shaps[chem_name] = dict(
            zip([FEATURE_LABELS.get(f,f) for f in feature_cols],
                mean_shap)
        )

        ax.barh(imp["label"], imp["mean_shap"],
                color=color, alpha=0.85, edgecolor="white")
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(f"{chem_name}\n(Top 10 features)",
                     fontweight="bold", color=color)
        ax.grid(True, alpha=0.3, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        "BatteryIQ — SHAP Feature Importance per Chemistry\n"
        "Which features drive SOH prediction for each chemistry?",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out = FIG_DIR / "fig35_shap_per_chemistry.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → fig35_shap_per_chemistry.png")

    # Print key differences
    print("\n   Key chemistry differences in feature importance:")
    common_feats = ["Cycle Number", "Discharge Capacity (Ah)",
                    "Normalised Capacity", "Capacity Fade Rate",
                    "Arrhenius Factor (exp(-Ea/RT))"]
    print(f"\n   {'Feature':<40} {'NMC':>8} {'LFP':>8} {'LiCoO2':>8}")
    print(f"   {'-'*68}")
    for feat in common_feats:
        vals = [all_mean_shaps[c].get(feat, 0) for c in chemistries.keys()]
        print(f"   {feat:<40} "
              f"{vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")


# ── Figure 5: Waterfall for one cell ──────────────────────────────────────
def plot_waterfall(model, df, feature_cols):
    print("\n📊 Figure 5: SHAP Waterfall (single prediction) ...")

    # Pick an interesting cell — one near EOL
    eol_cells = df[df["soh_pct"] < 85]["cell_id"].unique()
    if len(eol_cells) == 0:
        eol_cells = df["cell_id"].unique()

    cell_id  = eol_cells[0]
    cell_df  = df[df["cell_id"] == cell_id].sort_values("cycle_number")
    # Pick middle of degradation
    mid_idx  = len(cell_df) // 2
    row      = cell_df.iloc[mid_idx:mid_idx+1]
    X_row    = row[feature_cols].fillna(row[feature_cols].median())

    explainer  = shap.TreeExplainer(model)
    sv         = explainer.shap_values(X_row)[0]
    base       = explainer.expected_value
    pred       = model.predict(X_row)[0]
    actual_soh = row["soh_pct"].iloc[0]
    cycle_num  = row["cycle_number"].iloc[0]

    # Manual waterfall plot
    labels    = [FEATURE_LABELS.get(f, f) for f in feature_cols]
    shap_df   = pd.DataFrame({
        "feature": feature_cols,
        "label"  : labels,
        "shap"   : sv
    }).sort_values("shap", key=abs, ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors  = ["#EF4444" if v > 0 else "#378ADD"
               for v in shap_df["shap"]]
    bars    = ax.barh(shap_df["label"], shap_df["shap"],
                      color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, shap_df["shap"]):
        x_pos = bar.get_width() + (0.05 if val > 0 else -0.05)
        ha    = "left" if val > 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}%", va="center",
                ha=ha, fontsize=8, fontweight="bold")

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP Value (% impact on SOH prediction)")
    ax.set_title(
        f"BatteryIQ — SHAP Waterfall: {cell_id} at Cycle {int(cycle_num)}\n"
        f"Base prediction: {base:.1f}% | Final prediction: {pred:.1f}% | "
        f"Actual SOH: {actual_soh:.1f}%\n"
        f"Red = pushes SOH higher | Blue = pushes SOH lower",
        fontweight="bold", fontsize=10
    )
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add annotation box
    textstr = (f"Cell: {cell_id}\n"
               f"Cycle: {int(cycle_num)}\n"
               f"Actual SOH: {actual_soh:.1f}%\n"
               f"Predicted: {pred:.1f}%\n"
               f"Error: {abs(pred-actual_soh):.2f}%")
    props = dict(boxstyle="round", facecolor="#E1F5EE", alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment="bottom",
            horizontalalignment="right", bbox=props)

    plt.tight_layout()
    out = FIG_DIR / "fig36_shap_waterfall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → fig36_shap_waterfall.png")
    print(f"   Cell: {cell_id} | Cycle: {int(cycle_num)} | "
          f"Actual: {actual_soh:.1f}% | Predicted: {pred:.1f}%")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — SHAP Analysis: Publication-Quality Figures")
    print("="*60)

    # Load
    df, model, X, y, feature_cols = load_data_and_model()

    # Compute SHAP
    shap_values, X_sample, expected = compute_shap(
        model, X, feature_cols, n_sample=3000
    )

    # Generate all 5 figures
    imp_df = plot_shap_summary_bar(shap_values, feature_cols)
    plot_beeswarm(shap_values, X_sample, feature_cols)
    plot_dependence(shap_values, X_sample, feature_cols, imp_df)
    plot_shap_per_chemistry(model, df, feature_cols)
    plot_waterfall(model, df, feature_cols)

    # Save SHAP values for future use
    np.save(EXPL_DIR / "shap_values.npy", shap_values)
    X_sample.to_csv(EXPL_DIR / "shap_sample.csv", index=False)
    imp_df.to_csv(EXPL_DIR / "feature_importance.csv", index=False)

    print("\n" + "="*60)
    print("✅ SHAP Analysis Complete!")
    print("\n   Figures saved to memoire/figures/:")
    print("   fig32_shap_summary.png      ← Mean |SHAP| bar chart")
    print("   fig33_shap_beeswarm.png     ← Direction + magnitude")
    print("   fig34_shap_dependence.png   ← How features interact with SOH")
    print("   fig35_shap_per_chemistry.png ← NMC vs LFP vs LiCoO2")
    print("   fig36_shap_waterfall.png    ← Single prediction explained")
    print("\n   Data saved to ml/explainability/:")
    print("   shap_values.npy")
    print("   feature_importance.csv")
    print("\n   These 5 figures go in Chapter 5, Section 5.2 and 5.5")


if __name__ == "__main__":
    main()
