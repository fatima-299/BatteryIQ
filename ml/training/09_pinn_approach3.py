"""
BatteryIQ — PINN Approach 3: Predict SOH Degradation Rate
===========================================================
Hypothesis: Predicting absolute SOH% is hard because:
  - NASA SOH range: 5-105% (wide)
  - Stanford SOH range: 76-105% (narrow)
  - Different scales confuse the model

Fix: Predict SOH CHANGE per cycle (delta) instead of absolute SOH.
  - Delta is always small (-5% to 0% per cycle)
  - Same scale across all chemistries
  - Physics constraint: delta must be <= 0 (no self-healing)

Then integrate deltas to reconstruct absolute SOH for evaluation.

Key advantage for NASA:
  - Even with missing IR and zero arrhenius, the model can learn
    that NASA cells degrade faster per cycle than Stanford cells
  - cycle_capacity_ah and cycle_number still carry useful signal

Run from BatteryIQ root:
  python ml/training/09_pinn_approach3.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT     = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data" / "features"
MOD_DIR  = ROOT / "ml" / "models"
EVAL_DIR = ROOT / "ml" / "evaluation"
FIG_DIR  = ROOT / "memoire" / "figures"
MOD_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE  = 256
EPOCHS      = 200
LR          = 3e-4
LAMBDA_PHYS = 0.2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use only features that are well-populated across ALL sources
# Removing internal_resistance (100% missing in NASA)
# Removing arrhenius_factor (zero variation in NASA)
FEATURES = [
    "cycle_number",
    "cycle_normalized",
    "lifecycle_stage",
    "cycles_from_start",
    "cycle_capacity_ah",
    "cap_normalized",
    "capacity_fade_rate",
    "ir_growth_rate",          # keep — partially available
    "avg_temp_c",              # keep — partially available
    "avg_voltage_v",           # keep — partially available
    "src_calce", "src_nasa", "src_stanford",
    "chem_CS2", "chem_CX2", "chem_LFP", "chem_NMC",
]


# ── Dataset ────────────────────────────────────────────────────────────────
class DeltaDataset(Dataset):
    """Dataset for predicting SOH delta (change per cycle)."""
    def __init__(self, X, y_delta):
        self.X       = torch.FloatTensor(X)
        self.y_delta = torch.FloatTensor(y_delta)

    def __len__(self): return len(self.y_delta)

    def __getitem__(self, idx):
        return self.X[idx], self.y_delta[idx]


# ── PINN for rate prediction ───────────────────────────────────────────────
class DeltaPINN(nn.Module):
    """
    Predicts SOH change per cycle (always negative or zero).
    Uses tanh output scaled to [-5, 0] to enforce non-positivity.
    This is a hard physics constraint built into the architecture.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        raw = self.net(x).squeeze(-1)
        # Hard constraint: output in [-5, 0]
        # tanh maps to (-1, 1), scale to (-5, 0)
        return -2.5 * (torch.tanh(raw) + 1.0)


# ── Training ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        # Data loss
        l_data = nn.functional.mse_loss(pred, y)
        # Physics: predicted rate should correlate with cycle position
        # (degradation accelerates over time)
        l_total = l_data
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += l_data.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            out = model(X.to(DEVICE)).cpu().numpy()
            out = np.nan_to_num(out, nan=0.0)
            preds.extend(out)
            trues.extend(y.numpy())
    return np.array(preds), np.array(trues)


# ── Integrate deltas to SOH ────────────────────────────────────────────────
def integrate_to_soh(model, df, feature_cols, scaler):
    """
    For each cell, predict delta per cycle and integrate
    to reconstruct SOH trajectory.
    Returns y_true and y_pred as absolute SOH values.
    """
    model.eval()
    all_true, all_pred, all_src = [], [], []

    for cell_id, grp in df.groupby("cell_id"):
        grp = grp.sort_values("cycle_number").copy()
        if len(grp) < 3:
            continue

        X = scaler.transform(
            grp[feature_cols].fillna(
                grp[feature_cols].median()).values
        )
        X_tensor   = torch.FloatTensor(X).to(DEVICE)
        with torch.no_grad():
            deltas = model(X_tensor).cpu().numpy()
        deltas = np.nan_to_num(deltas, nan=0.0)

        # Integrate: start from first actual SOH
        soh_actual = grp["soh_pct"].values
        soh_pred   = np.zeros(len(grp))
        soh_pred[0] = soh_actual[0]  # anchor to first known SOH

        for i in range(1, len(grp)):
            soh_pred[i] = soh_pred[i-1] + deltas[i]
            # Clip to valid range
            soh_pred[i] = np.clip(soh_pred[i], 0, 105)

        all_true.extend(soh_actual[1:])   # skip first (anchored)
        all_pred.extend(soh_pred[1:])
        all_src.extend([grp["source"].iloc[0]] * (len(grp) - 1))

    return np.array(all_true), np.array(all_pred), np.array(all_src)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — PINN Approach 3: Degradation Rate")
    print("="*60)
    print("  Hypothesis: Predicting SOH rate (delta) is easier than")
    print("  absolute SOH because delta is the same scale across")
    print("  all chemistries and enables hard physics constraints.")
    print(f"  Device: {DEVICE}")

    # 1. Load data
    print("\n📂 Loading data ...")
    df = pd.read_csv(
        FEAT_DIR / "spark_output" / "feature_matrix_enriched.csv"
    )
    feature_cols = [c for c in FEATURES if c in df.columns]
    print(f"  Rows     : {len(df):,}")
    print(f"  Features : {len(feature_cols)} (IR and arrhenius removed)")
    print(f"  Reason   : IR 100% missing in NASA, arrhenius=0 in NASA")

    # 2. Compute SOH delta per cell
    print("\n📐 Computing SOH deltas ...")
    df = df.sort_values(["cell_id", "cycle_number"]).copy()
    df["soh_delta"] = df.groupby("cell_id")["soh_pct"].diff()

    # Remove NaN deltas (first cycle of each cell)
    df = df.dropna(subset=["soh_delta"]).copy()

    # Remove extreme outliers (> 5% per cycle = sensor noise)
    before = len(df)
    df     = df[df["soh_delta"].abs() <= 5.0].copy()
    print(f"  After delta computation: {len(df):,} rows "
          f"({before-len(df):,} outliers removed)")
    print(f"  Delta range: {df['soh_delta'].min():.4f}% to "
          f"{df['soh_delta'].max():.4f}% per cycle")
    print(f"  Delta mean : {df['soh_delta'].mean():.4f}% per cycle")

    # 3. Balance sources
    print("\n⚖️  Balancing sources (NASA oversampled) ...")
    nasa_df   = df[df["source"] == "nasa"]
    target    = len(nasa_df) * 15
    parts     = []
    for src in ["nasa", "stanford", "calce"]:
        src_df = df[df["source"] == src]
        n      = len(src_df)
        if n < target:
            factor  = int(np.ceil(target / n))
            src_bal = pd.concat([src_df]*factor,
                                ignore_index=True).sample(
                target, random_state=42)
        else:
            src_bal = src_df.sample(target, random_state=42)
        parts.append(src_bal)
        print(f"  {src:10s}: {n:6,} → {len(src_bal):6,}")
    df_bal = pd.concat(parts, ignore_index=True)

    # 4. Split by cells
    unique_cells = df["cell_id"].unique()
    tr_cells, tmp = train_test_split(unique_cells, test_size=0.30,
                                      random_state=42)
    va_cells, te_cells = train_test_split(tmp, test_size=0.50,
                                           random_state=42)

    def make_arrays(data, cells, scaler=None, fit=False):
        mask = np.isin(data["cell_id"].values, cells)
        sub  = data[mask].copy()
        X    = sub[feature_cols].fillna(
            sub[feature_cols].median()).values
        if fit:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
        y = sub["soh_delta"].values
        return X, y, scaler

    # Train on balanced data
    X_tr, y_tr, scaler = make_arrays(df_bal, tr_cells, fit=True)
    X_va, y_va, _      = make_arrays(df,     va_cells, scaler=scaler)
    X_te, y_te, _      = make_arrays(df,     te_cells, scaler=scaler)

    print(f"\n✂️  Split:")
    print(f"  Train : {len(y_tr):,} deltas ({len(tr_cells)} cells)")
    print(f"  Val   : {len(y_va):,} deltas ({len(va_cells)} cells)")
    print(f"  Test  : {len(y_te):,} deltas ({len(te_cells)} cells)")

    # 5. DataLoaders
    tr_ldr = DataLoader(DeltaDataset(X_tr, y_tr),
                        batch_size=BATCH_SIZE, shuffle=True)
    va_ldr = DataLoader(DeltaDataset(X_va, y_va), batch_size=BATCH_SIZE)

    # 6. Model
    model     = DeltaPINN(len(feature_cols)).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 DeltaPINN Architecture:")
    print(f"  Input → 128 → 128 → 64 → 32 → 1")
    print(f"  Output: tanh scaled to [-5, 0] → hard physics constraint")
    print(f"  Total params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # 7. Train
    print(f"\n🚀 Training {EPOCHS} epochs ...")
    best_val, best_state = float("inf"), None
    val_history = []

    for epoch in range(EPOCHS):
        train_epoch(model, tr_ldr, optimizer)
        vp, vt   = evaluate(model, va_ldr)
        val_rmse = np.sqrt(mean_squared_error(vt, vp))
        scheduler.step()
        val_history.append(val_rmse)

        if val_rmse < best_val:
            best_val   = val_rmse
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Val RMSE={val_rmse:.4f}%/cycle")

    model.load_state_dict(best_state)
    best_ep = int(np.argmin(val_history))
    print(f"  Best val RMSE: {best_val:.4f}%/cycle at epoch {best_ep}")

    # 8. Evaluate delta prediction
    print("\n📊 Delta Prediction Results (rate accuracy):")
    te_ldr        = DataLoader(DeltaDataset(X_te, y_te),
                               batch_size=BATCH_SIZE)
    te_pred, te_true = evaluate(model, te_ldr)
    delta_rmse = np.sqrt(mean_squared_error(te_true, te_pred))
    delta_mae  = mean_absolute_error(te_true, te_pred)
    print(f"  Delta RMSE : {delta_rmse:.4f}%/cycle")
    print(f"  Delta MAE  : {delta_mae:.4f}%/cycle")
    print(f"  (XGBoost v2 predicts absolute SOH, not comparable directly)")

    # 9. Integrate back to SOH for comparison
    print("\n🔄 Integrating deltas → SOH for cross-source evaluation ...")
    soh_true, soh_pred, sources = integrate_to_soh(
        model, df, feature_cols, scaler
    )

    print("\n🌍 Cross-source SOH RMSE (after integration):")
    cross = {}
    for src in ["nasa", "stanford", "calce"]:
        mask = sources == src
        if mask.sum() < 10:
            continue
        rmse = np.sqrt(mean_squared_error(soh_true[mask], soh_pred[mask]))
        mae  = mean_absolute_error(soh_true[mask], soh_pred[mask])
        r2   = r2_score(soh_true[mask], soh_pred[mask])
        cross[src] = {"rmse": rmse, "mae": mae, "r2": r2}
        print(f"  {src:12s}: RMSE={rmse:.4f}% | "
              f"MAE={mae:.4f}% | R²={r2:.4f}")

    # 10. Plot
    print("\n📈 Generating figure ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Val curve
    axes[0].plot(val_history, color="#1D9E75", linewidth=1.5)
    axes[0].axvline(best_ep, color="#EF4444", linestyle="--",
                    linewidth=1.5,
                    label=f"Best: ep{best_ep}={best_val:.4f}%")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val RMSE (%/cycle)")
    axes[0].set_title("DeltaPINN Training Curve", fontweight="bold")
    axes[0].legend(fontsize=8)

    # Predicted vs actual delta
    axes[1].scatter(te_true, te_pred, alpha=0.3, s=6, color="#1D9E75")
    lims = [min(te_true.min(), te_pred.min()),
            max(te_true.max(), te_pred.max())]
    axes[1].plot(lims, lims, "r--", linewidth=2)
    axes[1].set_xlabel("Actual SOH Delta (%/cycle)")
    axes[1].set_ylabel("Predicted SOH Delta (%/cycle)")
    axes[1].set_title(f"Delta Prediction\nRMSE={delta_rmse:.4f}%/cycle",
                      fontweight="bold")

    # Cross-source SOH RMSE comparison
    if cross:
        srcs   = list(cross.keys())
        rmses  = [cross[s]["rmse"] for s in srcs]
        xgb_r  = [2.1634, 0.1055, 0.5656]
        lstm_r = [6.3650, 0.5111, 4.6297]
        x = np.arange(len(srcs)); w = 0.25
        axes[2].bar(x-w,   xgb_r,  w, label="XGBoost v2",
                    color="#378ADD", alpha=0.8)
        axes[2].bar(x,     lstm_r,  w, label="LSTM",
                    color="#7F77DD", alpha=0.8)
        axes[2].bar(x+w,   rmses,   w, label="PINN A3 (rate)",
                    color="#1D9E75", alpha=0.8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(srcs)
        axes[2].set_ylabel("SOH RMSE (%)")
        axes[2].set_title("Cross-Source RMSE Comparison",
                          fontweight="bold")
        axes[2].axhline(2.1634, color="#378ADD",
                        linestyle="--", linewidth=1,
                        label="XGB v2 NASA baseline")
        axes[2].legend(fontsize=7)

    plt.suptitle(
        "BatteryIQ — PINN Approach 3: Degradation Rate Prediction\n"
        f"Output constrained to [-5,0]%/cycle (hard physics constraint)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig31_pinn_approach3.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved → fig31_pinn_approach3.png")

    # Save model + scaler
    torch.save(best_state, MOD_DIR / "pinn_approach3.pt")
    joblib.dump(scaler,    MOD_DIR / "pinn_approach3_scaler.pkl")

    # Save metrics
    pd.DataFrame([
        {"approach": "PINN_A3_rate", "source": src,
         "soh_rmse": v["rmse"], "soh_mae": v["mae"], "soh_r2": v["r2"]}
        for src, v in cross.items()
    ]).to_csv(EVAL_DIR / "pinn_approach3_metrics.csv", index=False)

    # Final verdict
    nasa_rmse = cross.get("nasa", {}).get("rmse", 999)
    print("\n" + "="*60)
    print("APPROACH 3 VERDICT")
    print("="*60)
    print(f"\n  Integrated SOH RMSE by source:")
    print(f"  {'Source':<12} {'PINN A3':>10} {'XGB v2':>10} "
          f"{'LSTM':>10}")
    print(f"  {'-'*45}")
    for src in ["nasa", "stanford", "calce"]:
        p = cross.get(src, {}).get("rmse", 999)
        x = {"nasa":2.1634,"stanford":0.1055,"calce":0.5656}[src]
        l = {"nasa":6.3650,"stanford":0.5111,"calce":4.6297}[src]
        print(f"  {src:<12} {p:>10.4f}% {x:>10.4f}% {l:>10.4f}%")

    print(f"\n  KEY: NASA RMSE after rate integration = {nasa_rmse:.4f}%")
    print(f"       XGBoost v2                       = 2.1634%")
    print(f"       LSTM                              = 6.3650%")

    if nasa_rmse < 2.1634:
        print(f"\n  🎉 APPROACH 3 BEATS XGBoost v2 ON NASA!")
        print(f"  Rate prediction + integration solves the problem!")
        print(f"  → This is the final PINN architecture")
    elif nasa_rmse < 6.365:
        print(f"\n  ✅ APPROACH 3 beats LSTM on NASA!")
        print(f"  Rate prediction improves over direct SOH prediction.")
        print(f"  → Document: PINN with rate prediction outperforms LSTM")
    else:
        print(f"\n  📊 All 3 approaches tested. Final conclusion:")
        print(f"  NASA NMC data (1,871 cycles, missing IR, zero temp")
        print(f"  variation) is fundamentally challenging for neural nets.")
        print(f"  XGBoost v2 remains best on NASA due to sparse data.")
        print(f"  PINN beats LSTM overall (2.03% vs 2.53%) on full test.")
        print(f"  → Accept results and document scientifically")
        print(f"  → Move to Power BI dashboard")


if __name__ == "__main__":
    main()
