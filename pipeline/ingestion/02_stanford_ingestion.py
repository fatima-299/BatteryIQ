"""
BatteryIQ — Stanford/MIT MATR Battery Dataset Ingestion v2
===========================================================
Fixed: cell_id now uses batch_date + cell_index (unique per cell)

Run from BatteryIQ root:
  python pipeline/ingestion/02_stanford_ingestion.py
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]
RAW_DIR  = ROOT / "data" / "raw" / "stanford"
PROC_DIR = ROOT / "data" / "processed"
PLOT_DIR = PROC_DIR / "plots"
PROC_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Process one batch file ─────────────────────────────────────────────────
def process_batch(mat_path: Path, batch_date: str) -> pd.DataFrame:
    print(f"\n📂 Processing {mat_path.name} ...")
    rows = []

    with h5py.File(mat_path, "r") as h:
        batch   = h["batch"]
        n_cells = batch["cycle_life"].shape[0]
        print(f"   Cells in batch: {n_cells}")

        for cell_idx in range(n_cells):

            # ── Unique cell ID: batch_date + cell index ────────────────────
            cell_id = f"{batch_date}_c{cell_idx:02d}"

            # ── Cycle life ─────────────────────────────────────────────────
            try:
                life_ref   = batch["cycle_life"][cell_idx, 0]
                cycle_life = int(h[life_ref][0, 0])
            except Exception:
                cycle_life = np.nan

            # ── Summary: cycle-level data ──────────────────────────────────
            try:
                sum_ref  = batch["summary"][cell_idx, 0]
                cell_sum = h[sum_ref]

                Qd      = cell_sum["QDischarge"][()].flatten()
                Tavg    = cell_sum["Tavg"][()].flatten()
                Tmax    = cell_sum["Tmax"][()].flatten()
                Tmin    = cell_sum["Tmin"][()].flatten()
                IR      = cell_sum["IR"][()].flatten()
                chgtime = cell_sum["chargetime"][()].flatten()
                cycles  = cell_sum["cycle"][()].flatten()
                n_cyc   = len(Qd)

            except Exception as e:
                print(f"   ⚠️  Cell {cell_idx} summary error: {e}")
                continue

            # Skip cycle 0 (initialisation, capacity = 0)
            start = 1 if (n_cyc > 1 and Qd[0] == 0) else 0

            # Nominal capacity = 95th percentile of valid discharge capacities
            valid_caps = Qd[Qd > 0.5]
            if len(valid_caps) == 0:
                continue
            nominal = float(np.percentile(valid_caps, 95))

            for i in range(start, n_cyc):
                cap = float(Qd[i]) if i < len(Qd) else np.nan
                if np.isnan(cap) or cap <= 0:
                    continue
                soh = round(cap / nominal * 100, 2)
                if soh > 105:       # filter artefacts
                    continue

                rows.append({
                    "cell_id"            : cell_id,
                    "batch"              : batch_date,
                    "source"             : "stanford",
                    "cycle_number"       : int(cycles[i]) if i < len(cycles) else i,
                    "cycle_life"         : cycle_life,
                    "cycle_capacity_ah"  : round(cap, 6),
                    "nominal_capacity_ah": round(nominal, 4),
                    "soh_pct"            : soh,
                    "avg_temp_c"         : float(Tavg[i])    if i < len(Tavg)    else np.nan,
                    "max_temp_c"         : float(Tmax[i])    if i < len(Tmax)    else np.nan,
                    "min_temp_c"         : float(Tmin[i])    if i < len(Tmin)    else np.nan,
                    "internal_resistance": float(IR[i])      if i < len(IR)      else np.nan,
                    "charge_time_s"      : float(chgtime[i]) if i < len(chgtime) else np.nan,
                    "status"             : "healthy" if soh >= 80 else "end_of_life",
                })

        print(f"   ✅ {len(rows)} cycle rows from {n_cells} cells")

    df = pd.concat([pd.DataFrame(rows)], ignore_index=True)
    df = df.dropna(subset=["soh_pct"]).copy()

    # Add RUL per cell
    rul_chunks = []
    for cid, grp in df.groupby("cell_id"):
        grp  = grp.sort_values("cycle_number").copy()
        eol  = grp[grp["soh_pct"] < 80]["cycle_number"].min()
        grp["rul_cycles"] = grp["cycle_number"].apply(
            lambda c: int(eol - c) if pd.notna(eol) and c < eol else 0
        )
        rul_chunks.append(grp)

    return pd.concat(rul_chunks, ignore_index=True)


# ── Plot SOH overview ──────────────────────────────────────────────────────
def plot_soh(df: pd.DataFrame):
    print("\n📈 Plotting SOH overview ...")
    batches = sorted(df["batch"].unique())
    fig, axes = plt.subplots(1, len(batches), figsize=(6 * len(batches), 5))
    if len(batches) == 1:
        axes = [axes]
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for ax, batch in zip(axes, batches):
        bdf   = df[df["batch"] == batch]
        cells = sorted(bdf["cell_id"].unique())
        for i, cell in enumerate(cells):
            cdf = bdf[bdf["cell_id"] == cell].sort_values("cycle_number")
            ax.plot(cdf["cycle_number"], cdf["soh_pct"],
                    linewidth=0.8, alpha=0.55, color=colors[i % 20])
        ax.axhline(80, color="#EF4444", linestyle="--", linewidth=1.5, label="EOL 80%")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("SOH (%)")
        ax.set_ylim(50, 108)
        ax.set_title(f"Batch: {batch}\n({len(cells)} cells)", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("BatteryIQ — Stanford/MIT MATR Dataset (3 Batches)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "stanford_soh_overview.png", dpi=150)
    plt.close()
    print(f"   ✅ Plot saved → stanford_soh_overview.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — Stanford/MIT MATR Ingestion v2")
    print("=" * 50)

    mat_files = sorted(RAW_DIR.glob("*.mat"))
    if not mat_files:
        print("⚠️  No .mat files found in data/raw/stanford/")
        return

    print(f"Found {len(mat_files)} batch files:")
    for f in mat_files:
        print(f"  {f.name} ({f.stat().st_size/1024/1024:.0f} MB)")

    all_batches = []
    for mat_file in mat_files:
        batch_date = mat_file.stem[:10]          # "2017-05-12"
        df_batch   = process_batch(mat_file, batch_date)
        all_batches.append(df_batch)

    combined = pd.concat(all_batches, ignore_index=True)

    # ── Quality report ─────────────────────────────────────────────────────
    print("\n🔍 Stanford Data Quality Report")
    print("=" * 50)
    print(f"  Total cycles       : {len(combined):,}")
    print(f"  Unique cells       : {combined['cell_id'].nunique()}")
    print(f"  Batches            : {combined['batch'].nunique()}")
    print(f"  SOH range          : {combined['soh_pct'].min():.1f}% → {combined['soh_pct'].max():.1f}%")
    print(f"  Healthy (≥80%)     : {(combined['status']=='healthy').sum():,}")
    print(f"  EOL (<80%)         : {(combined['status']=='end_of_life').sum():,}")
    print(f"  Avg cycle life     : {combined.groupby('cell_id')['cycle_life'].first().mean():.0f} cycles")
    print(f"  Cells sample       : {list(combined['cell_id'].unique()[:6])}")
    print(f"  Missing values     :")
    for col in ["soh_pct","avg_temp_c","internal_resistance","charge_time_s"]:
        print(f"    {col:30s}: {combined[col].isna().sum()}")

    out = PROC_DIR / "stanford_soh_per_cycle.csv"
    combined.to_csv(out, index=False)
    print(f"\n  ✅ Saved → {out}  ({len(combined):,} rows)")

    plot_soh(combined)
    print("\n✅ Stanford ingestion complete!")


if __name__ == "__main__":
    main()
