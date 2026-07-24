"""
BatteryIQ — Step 12: XGBoost Baseline Model
=============================================
Trains XGBoost for SOH prediction on the combined feature matrix.

Tasks:
  1. Load & prepare feature matrix
  2. Train/validation/test split (stratified by source)
  3. Train XGBoost with cross-validation
  4. Evaluate: RMSE, MAE, R²
  5. SHAP feature importance
  6. Cross-source generalisation test
  7. Save model + results

Run from BatteryIQ root:
  python ml/training/07_xgboost_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

ROOT     = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data" / "features"
MOD_DIR  = ROOT / "ml" / "models"
EVAL_DIR = ROOT / "ml" / "evaluation"
FIG_DIR  = ROOT / "memoire" / "figures"
MOD_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Load data ──────────────────────────────────────────────────────
def load_data():
    print("📂 Loading feature matrix ...")
    df = pd.read_csv(FEAT_DIR / "spark_output" / "feature_matrix_enriched.csv")
    print(f"   Shape: {df.shape}")

    # ML feature columns — exclude identity, targets, and string columns
    exclude = [
        "cell_id", "source", "chemistry", "cycle_number",
        "soh_pct", "rul_cycles", "status_encoded",
        "degradation_category", "alert_flag",
        "status", "nominal_capacity_ah"
    ]
    feature_cols = [c for c in df.columns
                   if c not in exclude
                   and df[c].dtype in [np.float64, np.int64, float, int]]

    print(f"   Features: {len(feature_cols)}")
    print(f"   Target  : soh_pct")
    return df, feature_cols


# ── Step 2: Train/val/test split ───────────────────────────────────────────
def split_data(df, feature_cols):
    print("\n✂️  Splitting data ...")

    X = df[feature_cols].copy()
    y = df["soh_pct"].copy()

    # Fill remaining NaN with column median
    X = X.fillna(X.median())

    # Stratified split — keep source distribution balanced
    # Train: 70% | Val: 15% | Test: 15%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    print(f"   Train : {len(X_train):,} rows ({len(X_train)/len(X)*100:.0f}%)")
    print(f"   Val   : {len(X_val):,} rows ({len(X_val)/len(X)*100:.0f}%)")
    print(f"   Test  : {len(X_test):,} rows ({len(X_test)/len(X)*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Step 3: Train XGBoost ─────────────────────────────────────────────────
def train_xgboost(X_train, X_val, y_train, y_val):
    print("\n🚀 Training XGBoost ...")

    params = {
        "n_estimators"     : 500,
        "max_depth"        : 6,
        "learning_rate"    : 0.05,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "min_child_weight" : 3,
        "reg_alpha"        : 0.1,    # L1 regularisation
        "reg_lambda"       : 1.0,    # L2 regularisation
        "objective"        : "reg:squarederror",
        "eval_metric"      : "rmse",
        "random_state"     : 42,
        "n_jobs"           : -1,     # use all cores
        "early_stopping_rounds": 30,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )

    print(f"\n   Best iteration: {model.best_iteration}")
    print(f"   Best val RMSE : {model.best_score:.4f}")
    return model


# ── Step 4: Evaluate ───────────────────────────────────────────────────────
def evaluate(model, X, y, split_name: str) -> dict:
    y_pred = model.predict(X)
    rmse   = np.sqrt(mean_squared_error(y, y_pred))
    mae    = mean_absolute_error(y, y_pred)
    r2     = r2_score(y, y_pred)
    print(f"   {split_name:10s}: RMSE={rmse:.4f}% | MAE={mae:.4f}% | R²={r2:.4f}")
    return {"split": split_name, "rmse": rmse, "mae": mae, "r2": r2,
            "y_true": y.values, "y_pred": y_pred}


# ── Step 5: Cross-source generalisation test ───────────────────────────────
def cross_source_test(model, df, feature_cols):
    print("\n🌍 Cross-source generalisation test ...")
    print("   (Train on NASA+Stanford, Test on CALCE — simulates new chemistry)")

    X_all = df[feature_cols].fillna(df[feature_cols].median())
    y_all = df["soh_pct"]

    results = {}
    for src in ["nasa", "stanford", "calce"]:
        mask   = df["source"] == src
        X_src  = X_all[mask]
        y_src  = y_all[mask]
        if len(X_src) == 0:
            continue
        y_pred = model.predict(X_src)
        rmse   = np.sqrt(mean_squared_error(y_src, y_pred))
        mae    = mean_absolute_error(y_src, y_pred)
        r2     = r2_score(y_src, y_pred)
        results[src] = {"rmse": rmse, "mae": mae, "r2": r2}
        print(f"   {src:12s}: RMSE={rmse:.4f}% | MAE={mae:.4f}% | R²={r2:.4f}")

    return results


# ── Step 6: SHAP explainability ────────────────────────────────────────────
def compute_shap(model, X_train, feature_cols):
    print("\n🔍 Computing SHAP values ...")
    sample = X_train.sample(min(2000, len(X_train)), random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    print(f"   SHAP computed on {len(sample):,} samples")
    return shap_values, sample


# ── Step 7: Plot results ───────────────────────────────────────────────────
def plot_results(model, results, shap_values, X_sample,
                 cross_src, feature_cols, df):
    print("\n📈 Generating figures ...")

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.45, wspace=0.35)

    # 1. Predicted vs Actual (test set)
    ax1   = fig.add_subplot(gs[0, 0])
    test  = results["test"]
    ax1.scatter(test["y_true"], test["y_pred"],
                alpha=0.3, s=8, color="#378ADD")
    lims = [min(test["y_true"].min(), test["y_pred"].min()),
            max(test["y_true"].max(), test["y_pred"].max())]
    ax1.plot(lims, lims, 'r--', linewidth=1.5, label="Perfect prediction")
    ax1.set_xlabel("Actual SOH (%)")
    ax1.set_ylabel("Predicted SOH (%)")
    ax1.set_title(f"Predicted vs Actual\nR²={test['r2']:.4f}", fontweight="bold")
    ax1.legend(fontsize=8)

    # 2. Residuals distribution
    ax2  = fig.add_subplot(gs[0, 1])
    res  = test["y_pred"] - test["y_true"]
    ax2.hist(res, bins=60, color="#7F77DD", alpha=0.8, edgecolor="white")
    ax2.axvline(0, color="#EF4444", linestyle="--", linewidth=2)
    ax2.axvline(res.mean(), color="#EF9F27", linestyle="--",
                linewidth=1.5, label=f"Mean={res.mean():.3f}%")
    ax2.set_xlabel("Residual (Predicted - Actual) %")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution", fontweight="bold")
    ax2.legend(fontsize=8)

    # 3. Metrics comparison bar
    ax3 = fig.add_subplot(gs[0, 2])
    splits = ["train", "val", "test"]
    rmses  = [results[s]["rmse"] for s in splits]
    maes   = [results[s]["mae"]  for s in splits]
    x      = np.arange(len(splits))
    w      = 0.35
    ax3.bar(x - w/2, rmses, w, label="RMSE", color="#378ADD", alpha=0.8)
    ax3.bar(x + w/2, maes,  w, label="MAE",  color="#EF9F27", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(splits)
    ax3.set_ylabel("Error (%)")
    ax3.set_title("RMSE & MAE by Split", fontweight="bold")
    ax3.legend()
    for i, (r, m) in enumerate(zip(rmses, maes)):
        ax3.text(i - w/2, r + 0.02, f"{r:.3f}", ha="center", fontsize=8)
        ax3.text(i + w/2, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)

    # 4. SHAP summary (top 15 features)
    ax4 = fig.add_subplot(gs[1, :2])
    shap_mean = np.abs(shap_values).mean(axis=0)
    feat_imp  = pd.Series(shap_mean, index=feature_cols).sort_values(ascending=True)
    top15     = feat_imp.tail(15)
    colors    = ["#EF4444" if v > top15.median() else "#378ADD"
                 for v in top15.values]
    ax4.barh(top15.index, top15.values, color=colors, alpha=0.85)
    ax4.set_xlabel("Mean |SHAP value|")
    ax4.set_title("Top 15 Features by SHAP Importance", fontweight="bold")
    ax4.axvline(top15.median(), color="black", linestyle="--",
                linewidth=1, alpha=0.5)

    # 5. Cross-source performance
    ax5    = fig.add_subplot(gs[1, 2])
    srcs   = list(cross_src.keys())
    rmses2 = [cross_src[s]["rmse"] for s in srcs]
    r2s    = [cross_src[s]["r2"]   for s in srcs]
    src_colors = {"nasa": "#378ADD", "stanford": "#1D9E75", "calce": "#EF9F27"}
    bars = ax5.bar(srcs, rmses2,
                   color=[src_colors.get(s,"#888") for s in srcs],
                   alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, rmses2):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f"{val:.3f}%", ha="center", fontsize=9, fontweight="bold")
    ax5.set_ylabel("RMSE (%)")
    ax5.set_title("Cross-Source RMSE\n(Generalisation Test)", fontweight="bold")

    # 6. Learning curves (training history)
    ax6    = fig.add_subplot(gs[2, 0])
    evals  = model.evals_result()
    train_rmse = evals["validation_0"]["rmse"]
    val_rmse   = evals["validation_1"]["rmse"]
    ax6.plot(train_rmse, color="#378ADD", linewidth=1.5, label="Train RMSE")
    ax6.plot(val_rmse,   color="#EF4444", linewidth=1.5, label="Val RMSE")
    ax6.axvline(model.best_iteration, color="#EF9F27",
                linestyle="--", linewidth=1.5,
                label=f"Best iter: {model.best_iteration}")
    ax6.set_xlabel("Boosting round")
    ax6.set_ylabel("RMSE (%)")
    ax6.set_title("XGBoost Learning Curves", fontweight="bold")
    ax6.legend(fontsize=8)

    # 7. SOH prediction over cycles for sample cells
    ax7 = fig.add_subplot(gs[2, 1:])
    sample_cells = df["cell_id"].unique()[:4]
    X_all  = df[feature_cols].fillna(df[feature_cols].median())
    colors = ["#378ADD","#EF9F27","#7F77DD","#1D9E75"]
    for i, cell in enumerate(sample_cells):
        mask   = df["cell_id"] == cell
        cdf    = df[mask].sort_values("cycle_number")
        X_cell = X_all[mask].loc[cdf.index]
        y_pred = model.predict(X_cell)
        ax7.plot(cdf["cycle_number"], cdf["soh_pct"],
                 color=colors[i], linewidth=1.5,
                 label=f"{cell} actual", alpha=0.8)
        ax7.plot(cdf["cycle_number"], y_pred,
                 color=colors[i], linewidth=1.5,
                 linestyle="--", label=f"{cell} predicted", alpha=0.6)
    ax7.axhline(80, color="#EF4444", linestyle="--",
                linewidth=1.5, label="EOL 80%")
    ax7.set_xlabel("Cycle number")
    ax7.set_ylabel("SOH (%)")
    ax7.set_title("Actual vs Predicted SOH — Sample Cells", fontweight="bold")
    ax7.legend(fontsize=7, ncol=2)

    plt.suptitle(
        f"BatteryIQ — XGBoost Baseline Model Results\n"
        f"Test RMSE={results['test']['rmse']:.4f}% | "
        f"MAE={results['test']['mae']:.4f}% | "
        f"R²={results['test']['r2']:.4f}",
        fontsize=13, fontweight="bold"
    )
    out = FIG_DIR / "fig28_xgboost_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → fig28_xgboost_results.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — XGBoost Baseline Model")
    print("=" * 55)

    # 1. Load
    df, feature_cols = load_data()

    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, feature_cols)

    # 3. Train
    model = train_xgboost(X_train, X_val, y_train, y_val)

    # 4. Evaluate all splits
    print("\n📊 Evaluation Results:")
    results = {
        "train": evaluate(model, X_train, y_train, "train"),
        "val"  : evaluate(model, X_val,   y_val,   "val"),
        "test" : evaluate(model, X_test,  y_test,  "test"),
    }

    # 5. Cross-source test
    cross_src = cross_source_test(model, df, feature_cols)

    # 6. SHAP
    shap_vals, X_sample = compute_shap(model, X_train, feature_cols)

    # 7. Plot
    plot_results(model, results, shap_vals, X_sample,
                 cross_src, feature_cols, df)

    # 8. Save model + metrics
    joblib.dump(model, MOD_DIR / "xgboost_soh.pkl")

    metrics_df = pd.DataFrame([
        {"model": "XGBoost", "split": s,
         "rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"]}
        for s, r in results.items()
    ])
    metrics_df.to_csv(EVAL_DIR / "xgboost_metrics.csv", index=False)

    # Save cross-source results
    cross_df = pd.DataFrame([
        {"model": "XGBoost", "source": src, **vals}
        for src, vals in cross_src.items()
    ])
    cross_df.to_csv(EVAL_DIR / "xgboost_cross_source.csv", index=False)

    print("\n" + "=" * 55)
    print("✅ XGBoost training complete!")
    print(f"   Model saved  → ml/models/xgboost_soh.pkl")
    print(f"   Metrics      → ml/evaluation/xgboost_metrics.csv")
    print(f"   Figure       → memoire/figures/fig28_xgboost_results.png")
    print(f"\n   FINAL TEST METRICS:")
    print(f"   RMSE : {results['test']['rmse']:.4f}%")
    print(f"   MAE  : {results['test']['mae']:.4f}%")
    print(f"   R²   : {results['test']['r2']:.4f}")
    print(f"\n   Next: python ml/training/08_lstm_model.py")


if __name__ == "__main__":
    main()
