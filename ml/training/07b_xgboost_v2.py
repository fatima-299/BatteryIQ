"""
BatteryIQ — XGBoost v2: Early Lifetime Prediction (No Lag Features)
=====================================================================
This version removes all lag, delta, and rolling features to simulate
the realistic scenario where we have LIMITED cycle history.

This is the correct baseline for comparing against LSTM and PINN.

Scenario: Given only static + physics features at cycle N,
predict SOH without knowing previous SOH values.

Features used:
  - cycle_number, cycle_capacity_ah, nominal_capacity_ah
  - internal_resistance, avg_temp_c, avg_voltage_v
  - capacity_fade_rate, ir_growth_rate, ir_cumulative_growth
  - cap_normalized, arrhenius_factor, cycle_normalized
  - lifecycle_stage, cycles_from_start
  - src_*, chem_* (source + chemistry encodings)

NOT used (removed):
  - soh_lag_*, soh_delta_*, soh_roll_*  ← these cause leakage
  - ir_roll_*, cap_lag_*, soh_acceleration

Run from BatteryIQ root:
  python ml/training/07b_xgboost_v2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap

ROOT     = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data" / "features"
MOD_DIR  = ROOT / "ml" / "models"
EVAL_DIR = ROOT / "ml" / "evaluation"
FIG_DIR  = ROOT / "memoire" / "figures"
MOD_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature selection — NO lags, NO rolling ────────────────────────────────
LEAKAGE_PATTERNS = [
    "soh_lag", "soh_delta", "soh_roll",
    "cap_lag", "ir_roll", "soh_acceleration",
    "cumulative_min_soh", "cycle_rank",
    "risk_score", "alert_flag", "degradation_category"
]

KEEP_FEATURES = [
    # Cycle position
    "cycle_number",
    "cycle_normalized",
    "lifecycle_stage",
    "cycles_from_start",

    # Raw measurements
    "cycle_capacity_ah",
    "avg_temp_c",
    "avg_voltage_v",
    "avg_current_a",
    "internal_resistance",

    # Physics-derived (computed from capacity trend, not SOH history)
    "capacity_fade_rate",
    "ir_growth_rate",
    "ir_cumulative_growth",
    "cap_normalized",
    "arrhenius_factor",

    # Source & chemistry identity
    "src_calce", "src_nasa", "src_stanford",
    "chem_CS2", "chem_CX2", "chem_LFP", "chem_NMC",
]


# ── Load data ──────────────────────────────────────────────────────────────
def load_data():
    print("📂 Loading feature matrix ...")
    df = pd.read_csv(
        FEAT_DIR / "spark_output" / "feature_matrix_enriched.csv"
    )
    print(f"   Total rows : {len(df):,}")

    # Keep only non-leakage features that exist in the dataframe
    feature_cols = [c for c in KEEP_FEATURES if c in df.columns]
    removed      = [c for c in KEEP_FEATURES if c not in df.columns]

    print(f"   Features   : {len(feature_cols)} (lag/rolling features removed)")
    if removed:
        print(f"   Not found  : {removed}")

    # Verify no leakage columns slipped through
    for col in feature_cols:
        for pattern in LEAKAGE_PATTERNS:
            if pattern in col:
                print(f"   ⚠️  WARNING: possible leakage in {col}")

    return df, feature_cols


# ── Split ──────────────────────────────────────────────────────────────────
def split_data(df, feature_cols):
    print("\n✂️  Splitting data ...")

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["soh_pct"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    print(f"   Train : {len(X_train):,} | Val : {len(X_val):,} | Test : {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Train ──────────────────────────────────────────────────────────────────
def train_xgboost_v2(X_train, X_val, y_train, y_val):
    print("\n🚀 Training XGBoost v2 (no lag features) ...")

    model = xgb.XGBRegressor(
        n_estimators          = 800,
        max_depth             = 7,
        learning_rate         = 0.03,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 5,
        reg_alpha             = 0.5,
        reg_lambda            = 2.0,
        objective             = "reg:squarederror",
        eval_metric           = "rmse",
        random_state          = 42,
        n_jobs                = -1,
        early_stopping_rounds = 50,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    print(f"\n   Best iteration : {model.best_iteration}")
    print(f"   Best val RMSE  : {model.best_score:.4f}%")
    return model


# ── Evaluate ───────────────────────────────────────────────────────────────
def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    rmse   = np.sqrt(mean_squared_error(y, y_pred))
    mae    = mean_absolute_error(y, y_pred)
    r2     = r2_score(y, y_pred)
    mape   = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
    print(f"   {name:10s}: RMSE={rmse:.4f}% | MAE={mae:.4f}% | "
          f"R²={r2:.4f} | MAPE={mape:.4f}%")
    return {"split": name, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
            "y_true": y.values, "y_pred": y_pred}


# ── Cross-source test ──────────────────────────────────────────────────────
def cross_source_test(model, df, feature_cols):
    print("\n🌍 Cross-source generalisation test ...")
    X_all = df[feature_cols].fillna(df[feature_cols].median())
    y_all = df["soh_pct"]
    results = {}
    for src in ["nasa", "stanford", "calce"]:
        mask   = df["source"] == src
        X_src  = X_all[mask]
        y_src  = y_all[mask]
        y_pred = model.predict(X_src)
        rmse   = np.sqrt(mean_squared_error(y_src, y_pred))
        mae    = mean_absolute_error(y_src, y_pred)
        r2     = r2_score(y_src, y_pred)
        mape   = np.mean(np.abs((y_src - y_pred) / (y_src + 1e-8))) * 100
        results[src] = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
        print(f"   {src:12s}: RMSE={rmse:.4f}% | MAE={mae:.4f}% | "
              f"R²={r2:.4f} | MAPE={mape:.4f}%")
    return results


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_results(model, results, df, feature_cols, cross_src):
    print("\n📈 Generating figures ...")

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Predicted vs Actual
    ax1  = fig.add_subplot(gs[0, 0])
    test = results["test"]
    ax1.scatter(test["y_true"], test["y_pred"],
                alpha=0.2, s=6, color="#378ADD")
    lims = [min(test["y_true"].min(), test["y_pred"].min()),
            max(test["y_true"].max(), test["y_pred"].max())]
    ax1.plot(lims, lims, "r--", linewidth=2, label="Perfect")
    ax1.set_xlabel("Actual SOH (%)")
    ax1.set_ylabel("Predicted SOH (%)")
    ax1.set_title(f"Predicted vs Actual\nR²={test['r2']:.4f}", fontweight="bold")
    ax1.legend(fontsize=8)

    # 2. Residuals
    ax2 = fig.add_subplot(gs[0, 1])
    res = test["y_pred"] - test["y_true"]
    ax2.hist(res, bins=60, color="#7F77DD", alpha=0.8, edgecolor="white")
    ax2.axvline(0, color="#EF4444", linestyle="--", linewidth=2)
    ax2.axvline(res.mean(), color="#EF9F27", linestyle="--",
                linewidth=1.5, label=f"Mean={res.mean():.3f}%")
    ax2.set_xlabel("Residual (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution", fontweight="bold")
    ax2.legend(fontsize=8)

    # 3. Metrics bar
    ax3 = fig.add_subplot(gs[0, 2])
    splits = ["train", "val", "test"]
    rmses  = [results[s]["rmse"] for s in splits]
    maes   = [results[s]["mae"]  for s in splits]
    x = np.arange(len(splits))
    w = 0.35
    ax3.bar(x - w/2, rmses, w, label="RMSE", color="#378ADD", alpha=0.8)
    ax3.bar(x + w/2, maes,  w, label="MAE",  color="#EF9F27", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(splits)
    ax3.set_ylabel("Error (%)")
    ax3.set_title("RMSE & MAE by Split", fontweight="bold")
    ax3.legend()
    for i, (r, m) in enumerate(zip(rmses, maes)):
        ax3.text(i-w/2, r+0.02, f"{r:.2f}", ha="center", fontsize=8)
        ax3.text(i+w/2, m+0.02, f"{m:.2f}", ha="center", fontsize=8)

    # 4. SHAP feature importance
    ax4 = fig.add_subplot(gs[1, :2])
    sample    = df[feature_cols].fillna(
        df[feature_cols].median()).sample(1000, random_state=42)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(sample)
    shap_mean  = np.abs(shap_vals).mean(axis=0)
    feat_imp   = pd.Series(shap_mean, index=feature_cols).sort_values()
    colors_bar = ["#EF4444" if v > feat_imp.median() else "#378ADD"
                  for v in feat_imp.values]
    ax4.barh(feat_imp.index, feat_imp.values, color=colors_bar, alpha=0.85)
    ax4.set_xlabel("Mean |SHAP value|")
    ax4.set_title("Feature Importance (SHAP) — No Lag Features",
                  fontweight="bold")

    # 5. Cross-source bar
    ax5    = fig.add_subplot(gs[1, 2])
    srcs   = list(cross_src.keys())
    rmses2 = [cross_src[s]["rmse"] for s in srcs]
    src_colors = {"nasa":"#378ADD","stanford":"#1D9E75","calce":"#EF9F27"}
    bars = ax5.bar(srcs, rmses2,
                   color=[src_colors.get(s,"#888") for s in srcs],
                   alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, rmses2):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"{val:.3f}%", ha="center", fontsize=9, fontweight="bold")
    ax5.set_ylabel("RMSE (%)")
    ax5.set_title("Cross-Source RMSE", fontweight="bold")

    # 6. Learning curves
    ax6 = fig.add_subplot(gs[2, 0])
    evals      = model.evals_result()
    train_rmse = evals["validation_0"]["rmse"]
    val_rmse   = evals["validation_1"]["rmse"]
    ax6.plot(train_rmse, color="#378ADD", linewidth=1.5, label="Train")
    ax6.plot(val_rmse,   color="#EF4444", linewidth=1.5, label="Val")
    ax6.axvline(model.best_iteration, color="#EF9F27", linestyle="--",
                linewidth=1.5, label=f"Best: {model.best_iteration}")
    ax6.set_xlabel("Boosting round")
    ax6.set_ylabel("RMSE (%)")
    ax6.set_title("Learning Curves", fontweight="bold")
    ax6.legend(fontsize=8)

    # 7. SOH prediction per cell sample
    ax7 = fig.add_subplot(gs[2, 1:])
    X_all = df[feature_cols].fillna(df[feature_cols].median())
    cells = df["cell_id"].unique()[:4]
    colors = ["#378ADD","#EF9F27","#7F77DD","#1D9E75"]
    for i, cell in enumerate(cells):
        mask   = df["cell_id"] == cell
        cdf    = df[mask].sort_values("cycle_number")
        y_pred = model.predict(X_all[mask].loc[cdf.index])
        ax7.plot(cdf["cycle_number"], cdf["soh_pct"],
                 color=colors[i], linewidth=1.5, label=f"{cell} actual")
        ax7.plot(cdf["cycle_number"], y_pred,
                 color=colors[i], linewidth=1.5,
                 linestyle="--", alpha=0.7, label=f"{cell} pred")
    ax7.axhline(80, color="#EF4444", linestyle="--", linewidth=1.5)
    ax7.set_xlabel("Cycle number")
    ax7.set_ylabel("SOH (%)")
    ax7.set_title("Actual vs Predicted — Sample Cells", fontweight="bold")
    ax7.legend(fontsize=7, ncol=2)

    plt.suptitle(
        f"BatteryIQ — XGBoost v2 (No Lag Features — Realistic Baseline)\n"
        f"Test RMSE={results['test']['rmse']:.4f}% | "
        f"MAE={results['test']['mae']:.4f}% | "
        f"R²={results['test']['r2']:.4f}",
        fontsize=13, fontweight="bold"
    )
    out = FIG_DIR / "fig29_xgboost_v2_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → fig29_xgboost_v2_results.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — XGBoost v2 (Early Prediction, No Lag Features)")
    print("=" * 60)

    df, feature_cols = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, feature_cols)

    model = train_xgboost_v2(X_train, X_val, y_train, y_val)

    print("\n📊 Evaluation Results:")
    results = {
        "train": evaluate(model, X_train, y_train, "train"),
        "val"  : evaluate(model, X_val,   y_val,   "val"),
        "test" : evaluate(model, X_test,  y_test,  "test"),
    }

    cross_src = cross_source_test(model, df, feature_cols)

    plot_results(model, results, df, feature_cols, cross_src)

    # Save
    joblib.dump(model, MOD_DIR / "xgboost_v2_soh.pkl")

    metrics_df = pd.DataFrame([
        {"model": "XGBoost_v2_no_lags", "split": s,
         "rmse": r["rmse"], "mae": r["mae"],
         "r2": r["r2"], "mape": r["mape"]}
        for s, r in results.items()
    ])
    metrics_df.to_csv(EVAL_DIR / "xgboost_v2_metrics.csv", index=False)

    cross_df = pd.DataFrame([
        {"model": "XGBoost_v2", "source": src, **vals}
        for src, vals in cross_src.items()
    ])
    cross_df.to_csv(EVAL_DIR / "xgboost_v2_cross_source.csv", index=False)

    print("\n" + "=" * 60)
    print("✅ XGBoost v2 complete!")
    print(f"\n   COMPARISON TABLE:")
    print(f"   {'Model':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"   {'-'*56}")
    print(f"   {'XGBoost v1 (with lags)':<30} {'0.1459':>8} {'0.0435':>8} {'0.9998':>8}")
    print(f"   {'XGBoost v2 (no lags)':<30} "
          f"{results['test']['rmse']:>8.4f} "
          f"{results['test']['mae']:>8.4f} "
          f"{results['test']['r2']:>8.4f}")
    print(f"\n   💡 v2 is the REALISTIC baseline for LSTM and PINN comparison")
    print(f"   Next: python ml/training/08_lstm_model.py")


if __name__ == "__main__":
    main()
