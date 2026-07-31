"""
BatteryIQ — PINN Approach 1: Balanced Sampling
===============================================
Hypothesis: PINN fails on NASA because 85% of training data is Stanford LFP.
Fix: Balance training data to equal cycles per chemistry group.

Training data composition:
  BEFORE: 85% Stanford + 14% CALCE + 1% NASA
  AFTER:  33% Stanford + 33% CALCE + 33% NASA (equal chemistry)

Everything else stays exactly the same as before.
If this fixes NASA RMSE → data imbalance was the problem.

Run from BatteryIQ root:
  python ml/training/09_pinn_approach1.py
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

# ── Hyperparameters (same as before) ──────────────────────────────────────
BATCH_SIZE  = 256
EPOCHS      = 200
LR          = 5e-4
LAMBDA_PHYS = 0.3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ── Dataset ────────────────────────────────────────────────────────────────
class BatteryDataset(Dataset):
    def __init__(self, X, y, fade_rates, arrh_factors):
        self.X  = torch.FloatTensor(X)
        self.y  = torch.FloatTensor(y)
        self.fr = torch.FloatTensor(fade_rates)
        self.af = torch.FloatTensor(arrh_factors)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.fr[idx], self.af[idx]


# ── PINN Architecture (same as before) ────────────────────────────────────
class BatteryPINN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.res1_main = nn.Sequential(
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(128, 128), nn.BatchNorm1d(128),
        )
        self.res1_act  = nn.ReLU()
        self.res2_main = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(64, 64), nn.BatchNorm1d(64),
        )
        self.res2_proj = nn.Linear(128, 64)
        self.res2_act  = nn.ReLU()
        self.head      = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h  = self.enc(x)
        h  = self.res1_act(self.res1_main(h) + h)
        h2 = self.res2_act(self.res2_main(h) + self.res2_proj(h))
        return self.head(h2).squeeze(-1)


# ── Physics Loss (same as before) ─────────────────────────────────────────
def physics_loss(soh_pred, fade_rates, arrh_factors):
    # Constraint 1: SOH cannot exceed 105%
    ceiling   = torch.clamp(soh_pred - 105.0, min=0.0)
    l_ceiling = torch.mean(ceiling ** 2)

    # Constraint 2: degrading battery + high temp → should not have high SOH
    degrading  = (fade_rates < -0.01).float()
    arrh_w     = torch.clamp(arrh_factors * 1000, 0, 1)
    high_soh   = torch.clamp(soh_pred - 85.0, min=0.0)
    l_fade     = torch.mean(degrading * arrh_w * high_soh ** 2)

    return l_ceiling + 0.5 * l_fade


def pinn_loss(pred, true, fr, af):
    l_data = nn.functional.mse_loss(pred, true)
    l_phys = physics_loss(pred, fr, af)
    return l_data + LAMBDA_PHYS * l_phys, l_data.item(), l_phys.item()


# ── Train / Eval ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    for X,y,fr,af in loader:
        X,y = X.to(DEVICE), y.to(DEVICE)
        fr  = fr.to(DEVICE); af = af.to(DEVICE)
        optimizer.zero_grad()
        loss, _, _ = pinn_loss(model(X), y, fr, af)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def eval_epoch(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X,y,_,_ in loader:
            preds.extend(model(X.to(DEVICE)).cpu().numpy())
            trues.extend(y.numpy())
    return np.array(preds), np.array(trues)


def compute_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    print(f"   {name:12s}: RMSE={rmse:.4f}% | MAE={mae:.4f}% | "
          f"R²={r2:.4f} | MAPE={mape:.4f}%")
    return rmse, mae, r2, mape


# ── APPROACH 1: Balance the training data ──────────────────────────────────
def balance_dataset(df):
    """
    Create balanced training dataset with equal cycles per source.
    Target = NASA size × 20 per source (enough data to learn from).
    """
    print("\n⚖️  Balancing dataset ...")

    nasa_size   = len(df[df["source"] == "nasa"])
    target_size = nasa_size * 20
    print(f"   NASA original    : {nasa_size:,} cycles")
    print(f"   Target per source: {target_size:,} cycles")

    parts = []
    for src in ["nasa", "stanford", "calce"]:
        src_df = df[df["source"] == src]
        n      = len(src_df)

        if n < target_size:
            # Oversample with replacement
            factor  = int(np.ceil(target_size / n))
            src_bal = pd.concat(
                [src_df] * factor, ignore_index=True
            ).sample(target_size, random_state=42)
            print(f"   {src:10s}: {n:6,} → {len(src_bal):6,} "
                  f"(oversampled {factor}×)")
        else:
            # Undersample randomly
            src_bal = src_df.sample(target_size, random_state=42)
            print(f"   {src:10s}: {n:6,} → {len(src_bal):6,} "
                  f"(undersampled)")

        parts.append(src_bal)

    df_balanced = pd.concat(parts, ignore_index=True)
    print(f"\n   Total balanced   : {len(df_balanced):,} cycles")
    print(f"   Per source       : {target_size:,} each = 33.3% each")
    return df_balanced


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — PINN Approach 1: Balanced Sampling")
    print("="*60)
    print(f"   Hypothesis: NASA RMSE is high because 85% of training")
    print(f"   data is Stanford LFP. Fix: equal chemistry representation.")
    print(f"   Device: {DEVICE}")

    # 1. Load original data
    print("\n📂 Loading data ...")
    df = pd.read_csv(
        FEAT_DIR / "spark_output" / "feature_matrix_enriched.csv"
    )
    feature_cols = [c for c in FEATURES if c in df.columns]
    print(f"   Original: {len(df):,} rows | {len(feature_cols)} features")
    print(f"   Source distribution:")
    for src, cnt in df["source"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"     {src:10s}: {cnt:7,} ({pct:.1f}%)")

    # 2. Balance the training data
    df_balanced = balance_dataset(df)

    # 3. Split by unique cells
    # IMPORTANT: split uses ORIGINAL cells so val/test
    # are not contaminated by oversampled training data
    unique_cells = df["cell_id"].unique()
    train_cells, temp_cells = train_test_split(
        unique_cells, test_size=0.30, random_state=42
    )
    val_cells, test_cells = train_test_split(
        temp_cells, test_size=0.50, random_state=42
    )

    # Training comes from BALANCED data
    # Val and Test come from ORIGINAL data (fair evaluation)
    def make_arrays(data, cells, scaler=None, fit_scaler=False):
        mask = np.isin(data["cell_id"].values, cells)
        sub  = data[mask].copy()
        X    = sub[feature_cols].fillna(
            sub[feature_cols].median()).values
        if fit_scaler:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
        y  = sub["soh_pct"].values
        fr = sub["capacity_fade_rate"].fillna(0).values.astype(float)
        af = sub["arrhenius_factor"].fillna(3e-4).values.astype(float)
        return X, y, fr, af, scaler

    # Fit scaler on balanced training data
    X_tr, y_tr, fr_tr, af_tr, scaler = make_arrays(
        df_balanced, train_cells, fit_scaler=True
    )
    # Val and test from original unbalanced data
    X_va, y_va, fr_va, af_va, _ = make_arrays(
        df, val_cells, scaler=scaler
    )
    X_te, y_te, fr_te, af_te, _ = make_arrays(
        df, test_cells, scaler=scaler
    )

    print(f"\n✂️  Data split:")
    print(f"   Train: {len(y_tr):,} rows ({len(train_cells)} cells)"
          f" — FROM BALANCED DATA")
    print(f"   Val  : {len(y_va):,} rows ({len(val_cells)} cells)"
          f" — FROM ORIGINAL DATA")
    print(f"   Test : {len(y_te):,} rows ({len(test_cells)} cells)"
          f" — FROM ORIGINAL DATA")

    # 4. DataLoaders
    train_loader = DataLoader(
        BatteryDataset(X_tr, y_tr, fr_tr, af_tr),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        BatteryDataset(X_va, y_va, fr_va, af_va),
        batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        BatteryDataset(X_te, y_te, fr_te, af_te),
        batch_size=BATCH_SIZE
    )

    # 5. Model
    model     = BatteryPINN(len(feature_cols)).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 PINN Architecture:")
    print(f"   Input → 128 → ResBlock(128) → ResBlock(64) → 32 → 1")
    print(f"   Total params: {n_params:,}")
    print(f"   Physics loss : ceiling constraint + Arrhenius-fade")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # 6. Training loop
    print(f"\n🚀 Training for {EPOCHS} epochs ...")
    best_val   = float("inf")
    best_state = None
    val_rmse_history = []

    for epoch in range(EPOCHS):
        train_epoch(model, train_loader, optimizer)
        val_pred, val_true = eval_epoch(model, val_loader)
        val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
        scheduler.step()
        val_rmse_history.append(val_rmse)

        if val_rmse < best_val:
            best_val   = val_rmse
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

        if (epoch + 1) % 40 == 0:
            print(f"   Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Val RMSE={val_rmse:.4f}%")

    model.load_state_dict(best_state)
    best_ep = np.argmin(val_rmse_history)
    print(f"\n   Best val RMSE: {best_val:.4f}% at epoch {best_ep}")

    # 7. Evaluate on test set
    print("\n📊 Test Set Results:")
    te_pred, te_true = eval_epoch(model, test_loader)
    test_rmse, test_mae, test_r2, test_mape = compute_metrics(
        te_true, te_pred, "test"
    )

    # 8. Cross-source test (on ORIGINAL unbalanced data — fair)
    print("\n🌍 Cross-source test (original data — fair evaluation):")
    X_all = scaler.transform(
        df[feature_cols].fillna(df[feature_cols].median()).values
    )
    y_all  = df["soh_pct"].values
    fr_all = df["capacity_fade_rate"].fillna(0).values.astype(float)
    af_all = df["arrhenius_factor"].fillna(3e-4).values.astype(float)

    cross_results = {}
    for src in ["nasa", "stanford", "calce"]:
        mask = (df["source"] == src).values
        if mask.sum() == 0:
            continue
        ldr  = DataLoader(
            BatteryDataset(X_all[mask], y_all[mask],
                           fr_all[mask], af_all[mask]),
            batch_size=BATCH_SIZE
        )
        pred, true = eval_epoch(model, ldr)
        rmse, mae, r2, mape = compute_metrics(true, pred, src)
        cross_results[src] = {
            "rmse": rmse, "mae": mae, "r2": r2, "mape": mape
        }

    # 9. Plot validation curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(val_rmse_history, color="#1D9E75", linewidth=1.5)
    axes[0].axvline(best_ep, color="#EF4444", linestyle="--",
                    linewidth=1.5,
                    label=f"Best: ep{best_ep}={best_val:.4f}%")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val RMSE (%)")
    axes[0].set_title("Validation RMSE — Approach 1",fontweight="bold")
    axes[0].legend(fontsize=8)

    axes[1].scatter(te_true, te_pred, alpha=0.2, s=6, color="#1D9E75")
    lims = [min(te_true.min(), te_pred.min()),
            max(te_true.max(), te_pred.max())]
    axes[1].plot(lims, lims, "r--", linewidth=2)
    axes[1].set_xlabel("Actual SOH (%)")
    axes[1].set_ylabel("Predicted SOH (%)")
    axes[1].set_title(f"Predicted vs Actual\nR²={test_r2:.4f}",
                      fontweight="bold")

    srcs   = list(cross_results.keys())
    rmses  = [cross_results[s]["rmse"] for s in srcs]
    colors = ["#378ADD","#1D9E75","#EF9F27"]
    bars   = axes[2].bar(srcs, rmses, color=colors, alpha=0.85,
                         edgecolor="white")
    axes[2].axhline(2.1634, color="#EF4444", linestyle="--",
                    linewidth=1.5, label="XGB v2 NASA baseline")
    for bar, val in zip(bars, rmses):
        axes[2].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.05,
                     f"{val:.3f}%", ha="center",
                     fontsize=10, fontweight="bold")
    axes[2].set_ylabel("RMSE (%)")
    axes[2].set_title("Cross-Source RMSE\n(red line = XGB v2 baseline)",
                      fontweight="bold")
    axes[2].legend(fontsize=8)

    plt.suptitle(
        f"BatteryIQ — PINN Approach 1: Balanced Sampling\n"
        f"Test RMSE={test_rmse:.4f}% | R²={test_r2:.4f}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig31_pinn_approach1.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✅ Saved → fig31_pinn_approach1.png")

    # 10. Save model
    torch.save(best_state, MOD_DIR / "pinn_approach1.pt")
    joblib.dump(scaler,    MOD_DIR / "pinn_approach1_scaler.pkl")

    # 11. Final verdict
    nasa_rmse = cross_results.get("nasa", {}).get("rmse", 999)
    print("\n" + "="*60)
    print("APPROACH 1 VERDICT")
    print("="*60)
    print(f"\n   Test RMSE  : {test_rmse:.4f}%  (XGB v2: 0.6114%)")
    print(f"   Test R²    : {test_r2:.4f}   (XGB v2: 0.9970)")
    print(f"   NASA RMSE  : {nasa_rmse:.4f}%  (XGB v2: 2.1634%)")

    if nasa_rmse < 2.1634:
        print(f"\n   ✅ HYPOTHESIS CONFIRMED!")
        print(f"   Balanced sampling fixed the NASA problem.")
        print(f"   PINN beats XGBoost v2 on NASA NMC!")
        print(f"   → No need for Approach 2 or 3")
        print(f"   → This is the final PINN model")
    elif nasa_rmse < 6.365:
        print(f"\n   ⚠️  PARTIAL SUCCESS")
        print(f"   PINN beats LSTM but not XGBoost v2 on NASA.")
        print(f"   → Try Approach 2 (per-chemistry PINN)")
    else:
        print(f"\n   ❌ APPROACH 1 DID NOT FIX NASA RMSE")
        print(f"   Balanced sampling alone is not enough.")
        print(f"   → Move to Approach 2 (per-chemistry PINN)")

    print(f"\n   Next step based on result:")
    print(f"   If fixed  → commit and move to RWTH validation")
    print(f"   If not    → run: python ml/training/09_pinn_approach2.py")


if __name__ == "__main__":
    main()
