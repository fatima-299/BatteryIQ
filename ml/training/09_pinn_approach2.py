"""
BatteryIQ — PINN Approach 2: Per-Chemistry PINN (fixed)
=========================================================
Fixed:
  - BatchNorm → LayerNorm (stable with small NMC dataset)
  - NaN guard on predictions
  - LR reduced to 1e-4 for stability
  - Gradient clipping increased

Run from BatteryIQ root:
  python ml/training/09_pinn_approach2.py
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

BATCH_SIZE  = 64     # small batch for small NMC dataset
EPOCHS      = 300
LR          = 1e-4   # reduced for stability
LAMBDA_PHYS = 0.1    # reduced physics weight
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

CHEMISTRY_GROUPS = {
    "NMC": "nasa",
    "LFP": "stanford",
    "LCO": "calce",
}

XGB_V2_RMSE = {"NMC": 2.1634, "LFP": 0.1055, "LCO": 0.5656}


# ── Dataset ────────────────────────────────────────────────────────────────
class BatteryDataset(Dataset):
    def __init__(self, X, y, fr, af):
        self.X  = torch.FloatTensor(X)
        self.y  = torch.FloatTensor(y)
        self.fr = torch.FloatTensor(fr)
        self.af = torch.FloatTensor(af)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.fr[idx], self.af[idx]


# ── PINN with LayerNorm (stable for small datasets) ────────────────────────
class BatteryPINN(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        # LayerNorm instead of BatchNorm — works with any batch size
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.res1_main = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.res1_act  = nn.ReLU()
        self.res2_main = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )
        self.res2_proj = nn.Linear(hidden, 64)
        self.res2_act  = nn.ReLU()
        self.head      = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Initialise weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h  = self.enc(x)
        h  = self.res1_act(self.res1_main(h) + h)
        h2 = self.res2_act(self.res2_main(h) + self.res2_proj(h))
        return self.head(h2).squeeze(-1)


# ── Physics Loss ───────────────────────────────────────────────────────────
def physics_loss(soh_pred, fr, af):
    # SOH ceiling — cannot exceed 105%
    l_ceil = torch.mean(torch.clamp(soh_pred - 105.0, min=0.0) ** 2)
    # Arrhenius-fade consistency
    deg    = (fr < -0.01).float()
    arrh_w = torch.clamp(af * 1000, 0, 1)
    high   = torch.clamp(soh_pred - 85.0, min=0.0)
    l_fade = torch.mean(deg * arrh_w * high ** 2)
    return l_ceil + 0.5 * l_fade


def loss_fn(pred, true, fr, af):
    l_d = nn.functional.mse_loss(pred, true)
    l_p = physics_loss(pred, fr, af)
    return l_d + LAMBDA_PHYS * l_p


# ── Train / Eval ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer):
    model.train()
    for X, y, fr, af in loader:
        X, y   = X.to(DEVICE), y.to(DEVICE)
        fr, af = fr.to(DEVICE), af.to(DEVICE)
        optimizer.zero_grad()
        pred   = model(X)
        loss   = loss_fn(pred, y, fr, af)
        if torch.isnan(loss):
            continue   # skip NaN batch
        loss.backward()
        # Aggressive gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y, _, _ in loader:
            out = model(X.to(DEVICE)).cpu().numpy()
            # NaN guard — replace with median SOH
            out = np.nan_to_num(out, nan=85.0, posinf=105.0, neginf=0.0)
            preds.extend(out)
            trues.extend(y.numpy())
    return np.array(preds), np.array(trues)


def compute_metrics(y_true, y_pred, name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    if name:
        print(f"   {name:12s}: RMSE={rmse:.4f}% | MAE={mae:.4f}% | "
              f"R²={r2:.4f} | MAPE={mape:.4f}%")
    return rmse, mae, r2, mape


# ── Train one chemistry PINN ───────────────────────────────────────────────
def train_chemistry_pinn(chem_name, chem_df, feature_cols):
    print(f"\n{'─'*55}")
    print(f"  Training PINN_{chem_name}")
    print(f"  Source: {CHEMISTRY_GROUPS[chem_name]} | "
          f"Cycles: {len(chem_df):,}")
    print(f"  SOH range: {chem_df['soh_pct'].min():.1f}% → "
          f"{chem_df['soh_pct'].max():.1f}%")
    print(f"{'─'*55}")

    cells = chem_df["cell_id"].unique()

    # Split strategy
    if len(cells) < 6:
        print(f"  Few cells ({len(cells)}) → row-based split")
        idx = np.arange(len(chem_df))
        tr_idx, tmp = train_test_split(idx, test_size=0.30,
                                        random_state=42)
        va_idx, te_idx = train_test_split(tmp, test_size=0.50,
                                           random_state=42)
        tr_df = chem_df.iloc[tr_idx]
        va_df = chem_df.iloc[va_idx]
        te_df = chem_df.iloc[te_idx]
    else:
        tr_cells, tmp = train_test_split(cells, test_size=0.30,
                                          random_state=42)
        va_cells, te_cells = train_test_split(tmp, test_size=0.50,
                                               random_state=42)
        tr_df = chem_df[np.isin(chem_df["cell_id"].values, tr_cells)]
        va_df = chem_df[np.isin(chem_df["cell_id"].values, va_cells)]
        te_df = chem_df[np.isin(chem_df["cell_id"].values, te_cells)]

    # Oversample small training sets
    if len(tr_df) < 3000:
        factor = int(np.ceil(3000 / len(tr_df)))
        tr_df  = pd.concat([tr_df] * factor,
                            ignore_index=True)
        print(f"  Oversampled training {factor}× → {len(tr_df):,} rows")

    print(f"  Train: {len(tr_df):,} | Val: {len(va_df):,} | "
          f"Test: {len(te_df):,}")

    # Prepare arrays
    scaler = StandardScaler()

    def to_arrays(data, fit=False):
        X  = data[feature_cols].fillna(
            data[feature_cols].median()).values
        X  = scaler.fit_transform(X) if fit else scaler.transform(X)
        y  = data["soh_pct"].values
        fr = data["capacity_fade_rate"].fillna(0).values.astype(float)
        af = data["arrhenius_factor"].fillna(3e-4).values.astype(float)
        return X, y, fr, af

    X_tr, y_tr, fr_tr, af_tr = to_arrays(tr_df, fit=True)
    X_va, y_va, fr_va, af_va = to_arrays(va_df)
    X_te, y_te, fr_te, af_te = to_arrays(te_df)

    def make_ldr(X, y, fr, af, shuffle=False):
        return DataLoader(
            BatteryDataset(X, y, fr, af),
            batch_size=BATCH_SIZE, shuffle=shuffle
        )

    tr_ldr = make_ldr(X_tr, y_tr, fr_tr, af_tr, shuffle=True)
    va_ldr = make_ldr(X_va, y_va, fr_va, af_va)
    te_ldr = make_ldr(X_te, y_te, fr_te, af_te)

    # Model
    model     = BatteryPINN(len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val, best_state = float("inf"), None
    val_history = []

    print(f"  Training {EPOCHS} epochs ...")
    for epoch in range(EPOCHS):
        train_one_epoch(model, tr_ldr, optimizer)
        vp, vt   = evaluate(model, va_ldr)
        val_rmse = np.sqrt(mean_squared_error(vt, vp))
        scheduler.step()
        val_history.append(val_rmse)

        if val_rmse < best_val:
            best_val   = val_rmse
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

        if (epoch + 1) % 75 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Val RMSE={val_rmse:.4f}%")

    model.load_state_dict(best_state)
    best_ep = int(np.argmin(val_history))
    print(f"  Best val RMSE: {best_val:.4f}% at epoch {best_ep}")

    # Test evaluation
    te_pred, te_true = evaluate(model, te_ldr)
    te_rmse, te_mae, te_r2, te_mape = compute_metrics(
        te_true, te_pred, f"PINN_{chem_name} test"
    )

    # Save
    torch.save(best_state, MOD_DIR / f"pinn_{chem_name}.pt")
    joblib.dump(scaler,    MOD_DIR / f"pinn_scaler_{chem_name}.pkl")

    return {
        "model"      : model,
        "scaler"     : scaler,
        "test_rmse"  : te_rmse,
        "test_mae"   : te_mae,
        "test_r2"    : te_r2,
        "val_history": val_history,
        "y_true"     : te_true,
        "y_pred"     : te_pred,
    }


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — PINN Approach 2: Per-Chemistry")
    print("="*60)
    print("  Hypothesis: Mixed-chemistry training causes interference.")
    print("  Fix: Dedicated PINN per chemistry group.")
    print(f"  Device: {DEVICE} | LR: {LR} | Norm: LayerNorm")

    # Load data
    print("\n📂 Loading data ...")
    df = pd.read_csv(
        FEAT_DIR / "spark_output" / "feature_matrix_enriched.csv"
    )
    feature_cols = [c for c in FEATURES if c in df.columns]

    source_to_chem = {"nasa": "NMC", "stanford": "LFP", "calce": "LCO"}
    df["chem_group"] = df["source"].map(source_to_chem)

    print(f"  Chemistry groups:")
    for chem, src in CHEMISTRY_GROUPS.items():
        n = (df["source"] == src).sum()
        print(f"    PINN_{chem}: {n:,} cycles from {src}")

    # Train per chemistry
    results = {}
    for chem_name in ["NMC", "LFP", "LCO"]:
        chem_df = df[df["chem_group"] == chem_name].copy()
        if len(chem_df) < 50:
            print(f"\n  ⚠️  Skipping {chem_name} — too few samples")
            continue
        results[chem_name] = train_chemistry_pinn(
            chem_name, chem_df, feature_cols
        )

    # Plot
    print("\n📈 Generating figure ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors    = {"NMC": "#378ADD", "LFP": "#1D9E75", "LCO": "#EF9F27"}

    # Val curves
    for chem, res in results.items():
        axes[0].plot(res["val_history"], color=colors[chem],
                     linewidth=1.5, label=f"PINN_{chem}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val RMSE (%)")
    axes[0].set_title("Val RMSE per Chemistry PINN", fontweight="bold")
    axes[0].legend()

    # RMSE comparison
    chem_names = list(results.keys())
    pinn_r     = [results[c]["test_rmse"] for c in chem_names]
    xgb_r      = [XGB_V2_RMSE.get(c, 0) for c in chem_names]
    x = np.arange(len(chem_names)); w = 0.35
    axes[1].bar(x - w/2, xgb_r,  w, label="XGBoost v2",
                color="#378ADD", alpha=0.8)
    axes[1].bar(x + w/2, pinn_r, w, label="PINN per-chem",
                color="#1D9E75", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(chem_names)
    axes[1].set_ylabel("Test RMSE (%)")
    axes[1].set_title("PINN vs XGBoost v2\nper Chemistry",
                      fontweight="bold")
    axes[1].legend()
    for i, (p, xv) in enumerate(zip(pinn_r, xgb_r)):
        axes[1].text(i+w/2, p+0.05, f"{p:.3f}%",
                     ha="center", fontsize=8, fontweight="bold")

    # NMC predicted vs actual
    if "NMC" in results:
        res = results["NMC"]
        axes[2].scatter(res["y_true"], res["y_pred"],
                        alpha=0.5, s=20, color="#378ADD")
        lims = [min(res["y_true"].min(), res["y_pred"].min()),
                max(res["y_true"].max(), res["y_pred"].max())]
        axes[2].plot(lims, lims, "r--", linewidth=2, label="Perfect")
        axes[2].set_xlabel("Actual SOH (%)")
        axes[2].set_ylabel("Predicted SOH (%)")
        axes[2].set_title(
            f"PINN_NMC: Predicted vs Actual\n"
            f"RMSE={results['NMC']['test_rmse']:.4f}%",
            fontweight="bold"
        )
        axes[2].legend()

    plt.suptitle(
        "BatteryIQ — PINN Approach 2: Per-Chemistry Models",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig31_pinn_approach2.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved → fig31_pinn_approach2.png")

    # Save metrics
    pd.DataFrame([
        {"approach": "PINN_A2", "chemistry": c,
         "test_rmse": r["test_rmse"], "test_r2": r["test_r2"]}
        for c, r in results.items()
    ]).to_csv(EVAL_DIR / "pinn_approach2_metrics.csv", index=False)

    # Final verdict
    print("\n" + "="*60)
    print("APPROACH 2 VERDICT")
    print("="*60)
    print(f"\n  {'Chemistry':<8} {'PINN RMSE':>12} "
          f"{'XGB v2':>12} {'Result':>12}")
    print(f"  {'-'*48}")

    for chem in chem_names:
        p = results[chem]["test_rmse"]
        x = XGB_V2_RMSE.get(chem, 0)
        result = "✅ BEATS XGB" if p < x else "❌ below XGB"
        print(f"  {chem:<8} {p:>12.4f}% {x:>12.4f}% {result:>12}")

    nmc_rmse = results.get("NMC", {}).get("test_rmse", 999)
    print(f"\n  KEY: PINN_NMC RMSE = {nmc_rmse:.4f}%")
    print(f"       XGBoost v2    = 2.1634%")
    print(f"       LSTM          = 6.3650%")

    if nmc_rmse < 2.1634:
        print(f"\n  🎉 SUCCESS! PINN_NMC BEATS XGBoost v2!")
        print(f"  Per-chemistry PINN is the winning architecture.")
        print(f"  → Save this as final PINN model")
        print(f"  → No need for Approach 3")
    elif nmc_rmse < 6.365:
        print(f"\n  ✅ PINN_NMC beats LSTM!")
        print(f"  Good result — document honestly.")
        print(f"  → Try Approach 3 for further improvement")
    else:
        print(f"\n  📊 NASA NMC dataset is too small for neural networks.")
        print(f"  XGBoost v2 remains best on NMC.")
        print(f"  → Try Approach 3 or accept current results")


if __name__ == "__main__":
    main()
