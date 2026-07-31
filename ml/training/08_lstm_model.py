"""
BatteryIQ — Step 13: LSTM Model for SOH Prediction
====================================================
Sequence-based SOH prediction using LSTM.

Approach:
  - Input  : sequence of last SEQ_LEN cycles (features without lags)
  - Output : SOH% at the next cycle
  - Architecture: 2-layer LSTM + dropout + dense head

This addresses XGBoost v2's weakness of ignoring temporal patterns.

Run from BatteryIQ root:
  python ml/training/08_lstm_model.py
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

# ── Hyperparameters ────────────────────────────────────────────────────────
SEQ_LEN    = 20      # look back 20 cycles
BATCH_SIZE = 256
EPOCHS     = 200
LR         = 0.001
HIDDEN_DIM = 64
N_LAYERS   = 2
DROPOUT    = 0.2
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same features as XGBoost v2 (no lags)
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
class BatterySequenceDataset(Dataset):
    """
    Creates sliding window sequences per cell.
    Input  : (SEQ_LEN, n_features)
    Target : SOH at cycle t+1
    """
    def __init__(self, sequences, targets):
        self.X = torch.FloatTensor(sequences)
        self.y = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── LSTM Model ─────────────────────────────────────────────────────────────
class BatteryLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.2):
        super(BatteryLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            dropout     = dropout if n_layers > 1 else 0,
            batch_first = True
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last time step output
        last_out    = lstm_out[:, -1, :]
        out         = self.dropout(last_out)
        return self.head(out).squeeze(-1)


# ── Build sequences ────────────────────────────────────────────────────────
def build_sequences(df, feature_cols, scaler=None, fit_scaler=False):
    """
    For each cell, create sliding window sequences.
    Returns X (n_sequences, SEQ_LEN, n_features), y (n_sequences,)
    """
    X_seqs, y_seqs, cell_ids = [], [], []

    # Scale features
    feat_data = df[feature_cols].fillna(df[feature_cols].median())
    if fit_scaler:
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_data)
    else:
        feat_scaled = scaler.transform(feat_data)

    feat_df = pd.DataFrame(feat_scaled,
                           columns=feature_cols,
                           index=df.index)

    for cell_id, group in df.groupby("cell_id"):
        group      = group.sort_values("cycle_number").copy()
        cell_feats = feat_df.loc[group.index].values
        cell_soh   = group["soh_pct"].values

        # Create sliding windows
        for i in range(SEQ_LEN, len(group)):
            seq    = cell_feats[i - SEQ_LEN:i]    # last SEQ_LEN cycles
            target = cell_soh[i]                   # SOH at cycle i
            X_seqs.append(seq)
            y_seqs.append(target)
            cell_ids.append(cell_id)

    return (np.array(X_seqs), np.array(y_seqs),
            np.array(cell_ids), scaler)


# ── Train one epoch ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss   = criterion(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


# ── Evaluate epoch ─────────────────────────────────────────────────────────
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_pred  = model(X_batch)
            loss    = criterion(y_pred, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_pred.extend(y_pred.cpu().numpy())
            all_true.extend(y_batch.cpu().numpy())
    return (total_loss / len(loader.dataset),
            np.array(all_pred), np.array(all_true))


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    print(f"   {name:10s}: RMSE={rmse:.4f}% | MAE={mae:.4f}% | "
          f"R²={r2:.4f} | MAPE={mape:.4f}%")
    return {"split": name, "rmse": rmse, "mae": mae,
            "r2": r2, "mape": mape,
            "y_true": y_true, "y_pred": y_pred}


# ── Plot results ───────────────────────────────────────────────────────────
def plot_results(results, history, df, model, scaler, feature_cols):
    print("\n📈 Generating figures ...")

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Predicted vs Actual
    ax1  = fig.add_subplot(gs[0, 0])
    test = results["test"]
    ax1.scatter(test["y_true"], test["y_pred"],
                alpha=0.2, s=6, color="#7F77DD")
    lims = [min(test["y_true"].min(), test["y_pred"].min()),
            max(test["y_true"].max(), test["y_pred"].max())]
    ax1.plot(lims, lims, "r--", linewidth=2, label="Perfect")
    ax1.set_xlabel("Actual SOH (%)")
    ax1.set_ylabel("Predicted SOH (%)")
    ax1.set_title(f"Predicted vs Actual\nR²={test['r2']:.4f}",
                  fontweight="bold")
    ax1.legend(fontsize=8)

    # 2. Residuals
    ax2 = fig.add_subplot(gs[0, 1])
    res = test["y_pred"] - test["y_true"]
    ax2.hist(res, bins=60, color="#1D9E75", alpha=0.8, edgecolor="white")
    ax2.axvline(0, color="#EF4444", linestyle="--", linewidth=2)
    ax2.axvline(res.mean(), color="#EF9F27", linestyle="--",
                linewidth=1.5, label=f"Mean={res.mean():.3f}%")
    ax2.set_xlabel("Residual (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution", fontweight="bold")
    ax2.legend(fontsize=8)

    # 3. Metrics comparison
    ax3 = fig.add_subplot(gs[0, 2])
    splits = ["train", "val", "test"]
    rmses  = [results[s]["rmse"] for s in splits]
    maes   = [results[s]["mae"]  for s in splits]
    x = np.arange(len(splits))
    w = 0.35
    ax3.bar(x-w/2, rmses, w, label="RMSE", color="#7F77DD", alpha=0.8)
    ax3.bar(x+w/2, maes,  w, label="MAE",  color="#EF9F27", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(splits)
    ax3.set_ylabel("Error (%)")
    ax3.set_title("RMSE & MAE by Split", fontweight="bold")
    ax3.legend()
    for i, (r, m) in enumerate(zip(rmses, maes)):
        ax3.text(i-w/2, r+0.005, f"{r:.3f}", ha="center", fontsize=8)
        ax3.text(i+w/2, m+0.005, f"{m:.3f}", ha="center", fontsize=8)

    # 4. Training curves
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(history["train_loss"], color="#7F77DD",
             linewidth=1.5, label="Train Loss (MSE)")
    ax4.plot(history["val_loss"],   color="#EF4444",
             linewidth=1.5, label="Val Loss (MSE)")
    best_ep = np.argmin(history["val_loss"])
    ax4.axvline(best_ep, color="#EF9F27", linestyle="--",
                linewidth=1.5, label=f"Best epoch: {best_ep}")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("MSE Loss")
    ax4.set_title("LSTM Training & Validation Loss", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.set_yscale("log")

    # 5. Val RMSE over epochs
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(history["val_rmse"], color="#1D9E75", linewidth=1.5)
    ax5.axhline(results["test"]["rmse"], color="#EF4444",
                linestyle="--", linewidth=1.5,
                label=f"Final test RMSE: {results['test']['rmse']:.4f}%")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Val RMSE (%)")
    ax5.set_title("Validation RMSE Over Training", fontweight="bold")
    ax5.legend(fontsize=8)

    # 6. Model comparison bar
    ax6 = fig.add_subplot(gs[2, 0])
    models  = ["XGBoost v1\n(lags)", "XGBoost v2\n(no lags)", "LSTM"]
    rmses_c = [0.1459, 0.6114, results["test"]["rmse"]]
    colors  = ["#EF9F27", "#378ADD", "#7F77DD"]
    bars    = ax6.bar(models, rmses_c, color=colors, alpha=0.85,
                      edgecolor="white")
    for bar, val in zip(bars, rmses_c):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.4f}%", ha="center", fontsize=9, fontweight="bold")
    ax6.set_ylabel("Test RMSE (%)")
    ax6.set_title("Model Comparison\n(RMSE — lower is better)",
                  fontweight="bold")

    # 7. Predicted trajectory for sample cells
    ax7 = fig.add_subplot(gs[2, 1:])
    sample_cells = df["cell_id"].unique()[:3]
    plot_colors  = ["#378ADD", "#EF9F27", "#7F77DD"]

    model.eval()
    feat_data   = df[feature_cols].fillna(df[feature_cols].median())
    feat_scaled = scaler.transform(feat_data)
    feat_df_sc  = pd.DataFrame(feat_scaled,
                               columns=feature_cols,
                               index=df.index)

    for i, cell in enumerate(sample_cells):
        grp        = df[df["cell_id"] == cell].sort_values("cycle_number")
        cell_feats = feat_df_sc.loc[grp.index].values
        cell_soh   = grp["soh_pct"].values
        cycles     = grp["cycle_number"].values

        if len(grp) <= SEQ_LEN:
            continue

        # Predict sequence by sequence
        preds = []
        for j in range(SEQ_LEN, len(grp)):
            seq     = torch.FloatTensor(
                cell_feats[j-SEQ_LEN:j]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(seq).item()
            preds.append(pred)

        ax7.plot(cycles, cell_soh,
                 color=plot_colors[i], linewidth=1.5,
                 label=f"{cell} actual")
        ax7.plot(cycles[SEQ_LEN:], preds,
                 color=plot_colors[i], linewidth=1.5,
                 linestyle="--", alpha=0.7,
                 label=f"{cell} LSTM pred")

    ax7.axhline(80, color="#EF4444", linestyle="--",
                linewidth=1.5, label="EOL 80%")
    ax7.set_xlabel("Cycle number")
    ax7.set_ylabel("SOH (%)")
    ax7.set_title("LSTM: Actual vs Predicted SOH", fontweight="bold")
    ax7.legend(fontsize=7, ncol=2)

    plt.suptitle(
        f"BatteryIQ — LSTM Model Results (seq_len={SEQ_LEN})\n"
        f"Test RMSE={results['test']['rmse']:.4f}% | "
        f"MAE={results['test']['mae']:.4f}% | "
        f"R²={results['test']['r2']:.4f}",
        fontsize=13, fontweight="bold"
    )
    out = FIG_DIR / "fig30_lstm_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → fig30_lstm_results.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"\n🔋 BatteryIQ — LSTM Model")
    print("=" * 55)
    print(f"   Device    : {DEVICE}")
    print(f"   Seq length: {SEQ_LEN} cycles")
    print(f"   Epochs    : {EPOCHS}")

    # 1. Load data
    print("\n📂 Loading data ...")
    df = pd.read_csv(
        FEAT_DIR / "spark_output" / "feature_matrix_enriched.csv"
    )
    feature_cols = [c for c in FEATURES if c in df.columns]
    print(f"   Rows     : {len(df):,}")
    print(f"   Features : {len(feature_cols)}")

    # 2. Build sequences
    print("\n🔢 Building sequences ...")
    X_all, y_all, cell_ids, scaler = build_sequences(
        df, feature_cols, fit_scaler=True
    )
    print(f"   Total sequences: {len(X_all):,}")
    print(f"   Sequence shape : {X_all[0].shape}")

    # 3. Split by unique cells (not random rows — avoid data leakage)
    unique_cells = df["cell_id"].unique()
    train_cells, temp_cells = train_test_split(
        unique_cells, test_size=0.30, random_state=42
    )
    val_cells, test_cells = train_test_split(
        temp_cells, test_size=0.50, random_state=42
    )

    train_mask = np.isin(cell_ids, train_cells)
    val_mask   = np.isin(cell_ids, val_cells)
    test_mask  = np.isin(cell_ids, test_cells)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    print(f"\n✂️  Split by cells (no leakage across cells):")
    print(f"   Train: {len(X_train):,} seqs ({len(train_cells)} cells)")
    print(f"   Val  : {len(X_val):,} seqs ({len(val_cells)} cells)")
    print(f"   Test : {len(X_test):,} seqs ({len(test_cells)} cells)")

    # 4. DataLoaders
    train_loader = DataLoader(
        BatterySequenceDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        BatterySequenceDataset(X_val, y_val),
        batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        BatterySequenceDataset(X_test, y_test),
        batch_size=BATCH_SIZE
    )

    # 5. Model
    model     = BatteryLSTM(
        input_dim  = len(feature_cols),
        hidden_dim = HIDDEN_DIM,
        n_layers   = N_LAYERS,
        dropout    = DROPOUT
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 LSTM Architecture:")
    print(f"   Input dim   : {len(feature_cols)}")
    print(f"   Hidden dim  : {HIDDEN_DIM}")
    print(f"   Layers      : {N_LAYERS}")
    print(f"   Total params: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
    )

    # 6. Training loop
    print(f"\n🚀 Training LSTM for {EPOCHS} epochs ...")
    history    = {"train_loss": [], "val_loss": [], "val_rmse": []}
    best_val   = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        train_loss              = train_epoch(model, train_loader,
                                             optimizer, criterion)
        val_loss, val_pred, val_true = eval_epoch(model, val_loader,
                                                   criterion)
        val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train MSE={train_loss:.4f} | "
                  f"Val MSE={val_loss:.4f} | "
                  f"Val RMSE={val_rmse:.4f}%")

    # Load best model
    model.load_state_dict(best_state)

    # 7. Final evaluation
    print("\n📊 Evaluation Results:")
    _, train_pred, train_true = eval_epoch(model, train_loader, criterion)
    _, val_pred,   val_true   = eval_epoch(model, val_loader,   criterion)
    _, test_pred,  test_true  = eval_epoch(model, test_loader,  criterion)

    results = {
        "train": compute_metrics(train_true, train_pred, "train"),
        "val"  : compute_metrics(val_true,   val_pred,   "val"),
        "test" : compute_metrics(test_true,  test_pred,  "test"),
    }

    # 8. Cross-source test
    print("\n🌍 Cross-source test ...")
    for src in ["nasa", "stanford", "calce"]:
        src_mask  = df["source"] == src
        src_cells = df[src_mask]["cell_id"].unique()
        src_seq_mask = np.isin(cell_ids, src_cells)
        if src_seq_mask.sum() == 0:
            continue
        X_src = X_all[src_seq_mask]
        y_src = y_all[src_seq_mask]
        src_loader = DataLoader(
            BatterySequenceDataset(X_src, y_src),
            batch_size=BATCH_SIZE
        )
        _, src_pred, src_true = eval_epoch(model, src_loader, criterion)
        rmse = np.sqrt(mean_squared_error(src_true, src_pred))
        mae  = mean_absolute_error(src_true, src_pred)
        r2   = r2_score(src_true, src_pred)
        print(f"   {src:12s}: RMSE={rmse:.4f}% | MAE={mae:.4f}% | R²={r2:.4f}")

    # 9. Plot
    plot_results(results, history, df, model, scaler, feature_cols)

    # 10. Save
    torch.save(best_state, MOD_DIR / "lstm_soh.pt")
    joblib.dump(scaler,    MOD_DIR / "lstm_scaler.pkl")

    metrics_df = pd.DataFrame([
        {"model": "LSTM", "split": s,
         "rmse": r["rmse"], "mae": r["mae"],
         "r2": r["r2"], "mape": r["mape"]}
        for s, r in results.items()
    ])
    metrics_df.to_csv(EVAL_DIR / "lstm_metrics.csv", index=False)

    print("\n" + "=" * 55)
    print("✅ LSTM training complete!")
    print(f"\n   UPDATED COMPARISON TABLE:")
    print(f"   {'Model':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"   {'-'*56}")
    print(f"   {'XGBoost v1 (lags)':<30} {'0.1459':>8} {'0.0435':>8} {'0.9998':>8}")
    print(f"   {'XGBoost v2 (no lags)':<30} {'0.6114':>8} {'0.1315':>8} {'0.9970':>8}")
    print(f"   {'LSTM':<30} "
          f"{results['test']['rmse']:>8.4f} "
          f"{results['test']['mae']:>8.4f} "
          f"{results['test']['r2']:>8.4f}")
    print(f"\n   Next: python ml/training/09_pinn_model.py")


if __name__ == "__main__":
    main()
