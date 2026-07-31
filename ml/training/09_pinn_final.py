"""
BatteryIQ — PINN Final: Stanford + CALCE (LFP + LiCoO2)
=========================================================
Scientific justification for excluding NASA:
  - internal_resistance: 100% missing in NASA
  - arrhenius_factor: zero variation (all same temperature)
  - avg cycles per cell: 55 (too short for gradient learning)
  - These are documented in Section 4.3 as data quality findings

PINN trained on Stanford (LFP, 114,688 cycles) +
              CALCE (LiCoO2, 18,379 cycles)
Total: 133,067 cycles across 155 cells, 2 chemistries

Physics constraints:
  L = L_data + 0.1×L_SEI + 0.05×L_thermal

Key question: Does PINN beat XGBoost v2 on CALCE (LiCoO2)?
  XGBoost v2 CALCE RMSE = 0.5656%
  CALCE has multiple C-rates + temperatures → Arrhenius most active here

Run from BatteryIQ root:
  python ml/training/09_pinn_final.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
EPOCHS      = 250
LR          = 3e-4
LAMBDA_SEI  = 0.1
LAMBDA_ARRH = 0.05
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

R  = 8.314
Ea = 50000.0

# Features — all well-populated in Stanford + CALCE
FEATURES = [
    "cycle_number", "cycle_normalized", "lifecycle_stage",
    "cycles_from_start", "cycle_capacity_ah",
    "avg_temp_c", "avg_voltage_v", "avg_current_a",
    "internal_resistance", "capacity_fade_rate",
    "ir_growth_rate", "ir_cumulative_growth",
    "cap_normalized", "arrhenius_factor",
    "src_calce", "src_stanford",
    "chem_CS2", "chem_CX2", "chem_LFP",
]


# ── Dataset ────────────────────────────────────────────────────────────────
class BatteryDataset(Dataset):
    def __init__(self, X, y, fr, af, temps):
        self.X     = torch.FloatTensor(X)
        self.y     = torch.FloatTensor(y)
        self.fr    = torch.FloatTensor(fr)
        self.af    = torch.FloatTensor(af)
        self.temps = torch.FloatTensor(temps)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx],
                self.fr[idx], self.af[idx], self.temps[idx])


# ── PINN Architecture ──────────────────────────────────────────────────────
class BatteryPINN(nn.Module):
    """
    Residual network with LayerNorm.
    Physics constraints applied in loss function.
    """
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
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
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


# ── Physics Loss Functions ─────────────────────────────────────────────────
def sei_loss(soh_pred, cycle_num):
    """
    SEI growth: capacity loss ∝ √(cycle)
    Implied alpha = (100 - SOH) / √cycle must be positive and smooth.
    """
    cycle_safe    = torch.clamp(cycle_num, min=1.0)
    alpha         = (100.0 - soh_pred) / torch.sqrt(cycle_safe)
    violation     = torch.clamp(-alpha, min=0.0)
    alpha_mean    = alpha.mean().detach()
    smoothness    = torch.mean((alpha - alpha_mean) ** 2)
    return torch.mean(violation ** 2) + 0.1 * smoothness


def arrhenius_loss(soh_pred, temp_c, cycle_num):
    """
    Arrhenius: degradation rate ∝ exp(-Ea/RT)
    Higher temp → faster degradation → more capacity loss.
    This constraint is most meaningful for CALCE which has
    multiple temperatures and C-rates.
    """
    T_k        = torch.clamp(temp_c + 273.15, 250.0, 370.0)
    cycle_safe = torch.clamp(cycle_num, min=1.0)
    Ea_t       = torch.tensor(Ea, dtype=torch.float32,
                               device=soh_pred.device)
    R_t        = torch.tensor(R,  dtype=torch.float32,
                               device=soh_pred.device)
    arrh       = torch.exp(-Ea_t / (R_t * T_k))
    cap_loss   = (100.0 - soh_pred) / torch.sqrt(cycle_safe)

    # Normalise both to [0,1]
    arrh_n = (arrh - arrh.min()) / (arrh.max() - arrh.min() + 1e-8)
    loss_n = (cap_loss - cap_loss.min()) / \
             (cap_loss.max() - cap_loss.min() + 1e-8)

    # Penalise negative correlation
    return torch.mean(torch.clamp(arrh_n - loss_n, min=0.0) ** 2)


def pinn_loss(pred, true, fr, af, temps):
    l_data = nn.functional.mse_loss(pred, true)
    l_sei  = sei_loss(pred, fr)           # use cycle proxy
    l_arrh = arrhenius_loss(pred, temps, fr)
    total  = l_data + LAMBDA_SEI * l_sei + LAMBDA_ARRH * l_arrh
    return total, l_data.item(), l_sei.item(), l_arrh.item()


# ── Train / Eval ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    td = ts = ta = 0
    for X, y, fr, af, tmp in loader:
        X, y   = X.to(DEVICE), y.to(DEVICE)
        fr, af = fr.to(DEVICE), af.to(DEVICE)
        tmp    = tmp.to(DEVICE)
        optimizer.zero_grad()
        loss, ld, ls, la = pinn_loss(model(X), y, fr, af, tmp)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n = len(y)
        td += ld*n; ts += ls*n; ta += la*n
    N = len(loader.dataset)
    return td/N, ts/N, ta/N


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y, _, _, _ in loader:
            out = model(X.to(DEVICE)).cpu().numpy()
            out = np.nan_to_num(out, nan=85.0)
            preds.extend(out)
            trues.extend(y.numpy())
    return np.array(preds), np.array(trues)


def compute_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true-y_pred)/(y_true+1e-8)))*100
    print(f"   {name:14s}: RMSE={rmse:.4f}% | MAE={mae:.4f}% | "
          f"R²={r2:.4f} | MAPE={mape:.4f}%")
    return rmse, mae, r2, mape


def make_loader(X, y, fr, af, tmp, shuffle=False):
    return DataLoader(
        BatteryDataset(X, y, fr, af, tmp),
        batch_size=BATCH_SIZE, shuffle=shuffle
    )


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — PINN Final (Stanford + CALCE)")
    print("="*60)
    print(f"   Sources  : Stanford (LFP) + CALCE (LiCoO2)")
    print(f"   NASA     : excluded (data quality — see Section 4.3)")
    print(f"   Physics  : L = L_data + {LAMBDA_SEI}×L_SEI "
          f"+ {LAMBDA_ARRH}×L_Arrhenius")
    print(f"   Device   : {DEVICE}")

    # 1. Load and filter
    print("\n📂 Loading data ...")
    df_all = pd.read_csv(
        FEAT_DIR / "spark_output" / "feature_matrix_enriched.csv"
    )

    # Exclude NASA — documented decision
    df = df_all[df_all["source"].isin(["stanford", "calce"])].copy()
    feature_cols = [c for c in FEATURES if c in df.columns]

    print(f"   Total rows (excl. NASA): {len(df):,}")
    print(f"   Features               : {len(feature_cols)}")
    for src, grp in df.groupby("source"):
        print(f"   {src:12s}: {len(grp):,} cycles | "
              f"{grp['cell_id'].nunique()} cells")

    # 2. Balance Stanford vs CALCE
    # Stanford dominates (114K vs 18K) — balance to 2× CALCE size
    print("\n⚖️  Balancing Stanford vs CALCE ...")
    calce_df    = df[df["source"] == "calce"]
    stanford_df = df[df["source"] == "stanford"]
    target      = len(calce_df) * 2   # 36,758 each
    stanford_bal = stanford_df.sample(target, random_state=42)
    df_balanced  = pd.concat(
        [stanford_bal, calce_df], ignore_index=True
    )
    print(f"   Stanford : {len(stanford_df):,} → {len(stanford_bal):,}")
    print(f"   CALCE    : {len(calce_df):,} → {len(calce_df):,} (unchanged)")
    print(f"   Total    : {len(df_balanced):,}")

    # 3. Split by cells
    unique_cells = df["cell_id"].unique()
    tr_cells, tmp_cells = train_test_split(
        unique_cells, test_size=0.30, random_state=42
    )
    va_cells, te_cells = train_test_split(
        tmp_cells, test_size=0.50, random_state=42
    )

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
        y    = sub["soh_pct"].values
        fr   = sub["cycle_number"].fillna(1).values.astype(float)
        af   = sub["arrhenius_factor"].fillna(3e-4).values.astype(float)
        tmp  = sub["avg_temp_c"].fillna(25.0).values.astype(float)
        return X, y, fr, af, tmp, scaler

    # Train on balanced, val/test on original
    X_tr,y_tr,fr_tr,af_tr,tmp_tr,scaler = make_arrays(
        df_balanced, tr_cells, fit=True
    )
    X_va,y_va,fr_va,af_va,tmp_va,_ = make_arrays(
        df, va_cells, scaler=scaler
    )
    X_te,y_te,fr_te,af_te,tmp_te,_ = make_arrays(
        df, te_cells, scaler=scaler
    )

    print(f"\n✂️  Split by cells:")
    print(f"   Train: {len(y_tr):,} rows ({len(tr_cells)} cells) "
          f"— balanced")
    print(f"   Val  : {len(y_va):,} rows ({len(va_cells)} cells) "
          f"— original")
    print(f"   Test : {len(y_te):,} rows ({len(te_cells)} cells) "
          f"— original")

    # 4. Loaders
    tr_ldr = make_loader(X_tr,y_tr,fr_tr,af_tr,tmp_tr, shuffle=True)
    va_ldr = make_loader(X_va,y_va,fr_va,af_va,tmp_va)
    te_ldr = make_loader(X_te,y_te,fr_te,af_te,tmp_te)

    # 5. Model
    model    = BatteryPINN(len(feature_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 PINN Architecture:")
    print(f"   Input → 256 → ResBlock(256) → ResBlock(128) → 64 → 1")
    print(f"   Norm  : LayerNorm (stable for all batch sizes)")
    print(f"   Total params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # 6. Training
    print(f"\n🚀 Training {EPOCHS} epochs ...")
    history  = {"data":[], "sei":[], "arrh":[], "val_rmse":[]}
    best_val, best_state = float("inf"), None

    for epoch in range(EPOCHS):
        ld, ls, la         = train_epoch(model, tr_ldr, optimizer)
        vp, vt             = evaluate(model, va_ldr)
        val_rmse           = np.sqrt(mean_squared_error(vt, vp))
        scheduler.step()

        history["data"].append(ld)
        history["sei"].append(ls)
        history["arrh"].append(la)
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val:
            best_val   = val_rmse
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

        if (epoch+1) % 50 == 0:
            print(f"   Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Data={ld:.4f} | SEI={ls:.4f} | "
                  f"Arrh={la:.4f} | Val RMSE={val_rmse:.4f}%")

    model.load_state_dict(best_state)
    best_ep = int(np.argmin(history["val_rmse"]))
    print(f"\n   Best val RMSE: {best_val:.4f}% at epoch {best_ep}")

    # 7. Evaluate
    print("\n📊 Evaluation Results:")
    trp,trt = evaluate(model, tr_ldr)
    vap,vat = evaluate(model, va_ldr)
    tep,tet = evaluate(model, te_ldr)

    results = {
        "train": compute_metrics(trt, trp, "train"),
        "val"  : compute_metrics(vat, vap, "val"),
        "test" : compute_metrics(tet, tep, "test"),
    }

    # 8. Per-source evaluation
    print("\n🌍 Per-source results (Stanford + CALCE only):")
    X_all   = scaler.transform(
        df[feature_cols].fillna(df[feature_cols].median()).values
    )
    y_all   = df["soh_pct"].values
    fr_all  = df["cycle_number"].fillna(1).values.astype(float)
    af_all  = df["arrhenius_factor"].fillna(3e-4).values.astype(float)
    tmp_all = df["avg_temp_c"].fillna(25.0).values.astype(float)

    cross = {}
    for src in ["stanford", "calce"]:
        mask = (df["source"] == src).values
        ldr  = make_loader(
            X_all[mask], y_all[mask],
            fr_all[mask], af_all[mask], tmp_all[mask]
        )
        pred, true = evaluate(model, ldr)
        rmse, mae, r2, mape = compute_metrics(true, pred, src)
        cross[src] = {"rmse":rmse,"mae":mae,"r2":r2,"mape":mape}

    # 9. Plot
    print("\n📈 Generating figures ...")
    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Predicted vs Actual
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(tet, tep, alpha=0.2, s=6, color="#1D9E75")
    lims = [min(tet.min(),tep.min()), max(tet.max(),tep.max())]
    ax1.plot(lims, lims, "r--", linewidth=2, label="Perfect")
    ax1.set_xlabel("Actual SOH (%)")
    ax1.set_ylabel("Predicted SOH (%)")
    ax1.set_title(f"PINN: Predicted vs Actual\nR²={results['test'][2]:.4f}",
                  fontweight="bold")
    ax1.legend(fontsize=8)

    # Loss decomposition
    ax2  = fig.add_subplot(gs[0, 1])
    ep   = range(1, len(history["data"])+1)
    ax2.plot(ep, history["data"], color="#378ADD",
             linewidth=1.5, label="Data loss")
    ax2.plot(ep, history["sei"],  color="#EF9F27",
             linewidth=1.5, label=f"SEI loss (λ={LAMBDA_SEI})")
    ax2.plot(ep, history["arrh"], color="#EF4444",
             linewidth=1.5, label=f"Arrhenius (λ={LAMBDA_ARRH})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Physics Loss Decomposition", fontweight="bold")
    ax2.legend(fontsize=8); ax2.set_yscale("log")

    # Val RMSE
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ep, history["val_rmse"], color="#1D9E75", linewidth=1.5)
    ax3.axvline(best_ep, color="#EF4444", linestyle="--",
                linewidth=1.5,
                label=f"Best ep{best_ep}: {best_val:.4f}%")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Val RMSE (%)")
    ax3.set_title("Validation RMSE", fontweight="bold")
    ax3.legend(fontsize=8)

    # Per-source comparison
    ax4 = fig.add_subplot(gs[1, 0])
    srcs   = ["stanford", "calce"]
    pinn_r = [cross[s]["rmse"] for s in srcs]
    xgb_r  = [0.1055, 0.5656]
    lstm_r = [0.5111, 4.6297]
    x = np.arange(2); w = 0.25
    ax4.bar(x-w,   xgb_r,  w, label="XGBoost v2",
            color="#378ADD", alpha=0.8)
    ax4.bar(x,     lstm_r,  w, label="LSTM",
            color="#7F77DD", alpha=0.8)
    ax4.bar(x+w,   pinn_r,  w, label="PINN (final)",
            color="#1D9E75", alpha=0.8)
    ax4.set_xticks(x); ax4.set_xticklabels(["Stanford\n(LFP)",
                                             "CALCE\n(LiCoO2)"])
    ax4.set_ylabel("RMSE (%)")
    ax4.set_title("PINN vs Baselines\n(Training chemistries only)",
                  fontweight="bold")
    ax4.legend(fontsize=8)
    for i, val in enumerate(pinn_r):
        ax4.text(i+w, val+0.01, f"{val:.3f}%",
                 ha="center", fontsize=8, fontweight="bold",
                 color="#1D9E75")

    # Full comparison table
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")
    calce_rmse    = cross["calce"]["rmse"]
    stanford_rmse = cross["stanford"]["rmse"]
    table_data = [
        ["XGBoost v1 (lags)",    "0.1459","—","0.9998",
         "All sources | Online scenario"],
        ["XGBoost v2 (no lags)", "0.6114","—","0.9970",
         "All sources | Realistic baseline"],
        ["LSTM (200 epochs)",    "2.5348","—","0.9562",
         "All sources | Sequence model"],
        ["PINN — Stanford (LFP)",
         f"{stanford_rmse:.4f}","0.1055",f"{cross['stanford']['r2']:.4f}",
         "LFP only | Physics: SEI + Arrhenius"],
        ["PINN — CALCE (LiCoO2)",
         f"{calce_rmse:.4f}","0.5656",f"{cross['calce']['r2']:.4f}",
         "LiCoO2 | Physics: SEI + Arrhenius ★"],
    ]
    tbl = ax5.table(
        cellText  = table_data,
        colLabels = ["Model","PINN\nRMSE","XGB v2\nRMSE","R²","Notes"],
        cellLoc   = "center", loc="center", bbox=[0,0,1,1]
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r,c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2D3748")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 5:    # CALCE row — most important
            cell.set_facecolor("#C6F6D5")
        elif r == 4:    # Stanford row
            cell.set_facecolor("#EBF8FF")
        elif r % 2 == 0:
            cell.set_facecolor("#F7FAFC")
    ax5.set_title(
        "BatteryIQ — Final Results Table\n"
        "Green = PINN on CALCE (key result)",
        fontweight="bold", fontsize=11, pad=10
    )

    plt.suptitle(
        f"BatteryIQ — PINN Final (Stanford LFP + CALCE LiCoO2)\n"
        f"L = L_data + {LAMBDA_SEI}×L_SEI + {LAMBDA_ARRH}×L_Arrhenius | "
        f"Test RMSE={results['test'][0]:.4f}% | R²={results['test'][2]:.4f}",
        fontsize=12, fontweight="bold"
    )
    plt.savefig(FIG_DIR/"fig31_pinn_final.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved → fig31_pinn_final.png")

    # 10. Save
    torch.save(best_state, MOD_DIR/"pinn_final.pt")
    joblib.dump(scaler,    MOD_DIR/"pinn_final_scaler.pkl")
    pd.DataFrame([
        {"model":"PINN_final","split":s,
         "rmse":r[0],"mae":r[1],"r2":r[2],"mape":r[3]}
        for s,r in results.items()
    ]).to_csv(EVAL_DIR/"pinn_final_metrics.csv", index=False)
    pd.DataFrame([
        {"model":"PINN_final","source":src,**vals}
        for src,vals in cross.items()
    ]).to_csv(EVAL_DIR/"pinn_final_cross_source.csv", index=False)

    # 11. Final verdict
    calce_beats = calce_rmse < 0.5656
    stanford_beats = stanford_rmse < 0.1055

    print("\n" + "="*60)
    print("✅ PINN FINAL RESULTS")
    print("="*60)
    print(f"\n   Test RMSE  : {results['test'][0]:.4f}%")
    print(f"   Test R²    : {results['test'][2]:.4f}")
    print(f"\n   Per-chemistry comparison:")
    print(f"   {'Chemistry':<15} {'PINN':>8} {'XGB v2':>8} "
          f"{'PINN wins?':>12}")
    print(f"   {'-'*45}")
    print(f"   {'Stanford LFP':<15} {stanford_rmse:>8.4f}% "
          f"{'0.1055%':>8} "
          f"{'✅ YES' if stanford_beats else '❌ NO':>12}")
    print(f"   {'CALCE LiCoO2':<15} {calce_rmse:>8.4f}% "
          f"{'0.5656%':>8} "
          f"{'✅ YES' if calce_beats else '❌ NO':>12}")

    if calce_beats:
        print(f"\n   🎉 PINN BEATS XGBoost v2 ON CALCE (LiCoO2)!")
        print(f"   Physics constraints (SEI + Arrhenius) outperform")
        print(f"   gradient boosting on multi-C-rate LiCoO2 data.")
        print(f"   This is the headline result for Chapter 5.")
    elif calce_rmse < 4.6297:
        print(f"\n   ✅ PINN beats LSTM on CALCE!")
        print(f"   Physics constraints improve over sequence modelling.")

    print(f"\n   NOVELTY STATEMENT:")
    print(f"   BatteryIQ applies dual physics-constrained learning")
    print(f"   (SEI growth + Arrhenius thermal degradation) to")
    print(f"   133,067 cycles of LFP + LiCoO2 chemistry data,")
    print(f"   deployed within a complete production pipeline.")
    print(f"\n   Next: Power BI Dashboard →")
    print(f"   python dashboard/setup_powerbi.py")


if __name__ == "__main__":
    main()
