"""
BatteryIQ — Step 4: Combine All Datasets into Unified Feature Table
====================================================================
Combines NASA + Stanford + CALCE into one master CSV.

Input files (data/processed/):
  nasa_soh_per_cycle.csv
  stanford_soh_per_cycle.csv
  calce_soh_per_cycle.csv

Output (data/features/):
  combined_soh.csv          — unified SOH table, all sources
  combined_soh_clean.csv    — filtered, validated, ML-ready

Run from BatteryIQ root:
  python pipeline/etl/04_combine_datasets.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
PROC_DIR  = ROOT / "data" / "processed"
FEAT_DIR  = ROOT / "data" / "features"
PLOT_DIR  = PROC_DIR / "plots"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# ── Unified schema ─────────────────────────────────────────────────────────
# Every dataset gets mapped to these columns:
SCHEMA = [
    "cell_id",           # unique cell identifier
    "source",            # nasa / stanford / calce
    "chemistry",         # NMC / LFP / LCO / CS2 / CX2
    "cycle_number",      # global cycle number per cell
    "cycle_capacity_ah", # discharge capacity this cycle
    "nominal_capacity_ah",
    "soh_pct",           # State of Health %
    "rul_cycles",        # Remaining Useful Life in cycles
    "status",            # healthy / end_of_life
    "avg_voltage_v",
    "avg_current_a",
    "avg_temp_c",
    "internal_resistance",
]


# ── Load & normalise NASA ──────────────────────────────────────────────────
def load_nasa() -> pd.DataFrame:
    print("📂 Loading NASA ...")
    df = pd.read_csv(PROC_DIR / "nasa_soh_per_cycle.csv")

    # Add chemistry info (NASA uses 18650 Li-ion = NMC/LCO)
    df = df.rename(columns={"battery_id": "cell_id", "Re": "internal_resistance"})
    df["chemistry"] = "NMC"
    df["source"]    = "nasa"

    # Map column names to schema
    rename = {
        "avg_voltage_v" : "avg_voltage_v",
        "avg_current_a" : "avg_current_a",
        "avg_temp_c"    : "avg_temp_c",
        "Re"            : "internal_resistance",   # use Re as IR proxy
    }
    df = df.rename(columns={k: v for k, v in rename.items()
                            if k in df.columns and k != v})

    # If internal_resistance not present, use Re
    if "internal_resistance" not in df.columns and "Re" in df.columns:
        df["internal_resistance"] = df["Re"]

    print(f"   ✅ {len(df):,} cycles | {df['cell_id'].nunique()} cells")
    return df


# ── Load & normalise Stanford ──────────────────────────────────────────────
def load_stanford() -> pd.DataFrame:
    print("📂 Loading Stanford ...")
    df = pd.read_csv(PROC_DIR / "stanford_soh_per_cycle.csv")

    df["chemistry"] = "LFP"   # Stanford uses LFP/Graphite cells
    df["source"]    = "stanford"

    # Stanford has internal_resistance column directly
    rename = {
        "internal_resistance": "internal_resistance",
        "avg_temp_c"         : "avg_temp_c",
    }

    # avg_voltage_v and avg_current_a may not exist in stanford
    if "avg_voltage_v" not in df.columns:
        df["avg_voltage_v"] = np.nan
    if "avg_current_a" not in df.columns:
        df["avg_current_a"] = np.nan
    if "avg_temp_c" not in df.columns and "avg_temp_c" not in df.columns:
        df["avg_temp_c"] = np.nan

    print(f"   ✅ {len(df):,} cycles | {df['cell_id'].nunique()} cells")
    return df


# ── Load & normalise CALCE ─────────────────────────────────────────────────
def load_calce() -> pd.DataFrame:
    print("📂 Loading CALCE ...")
    df = pd.read_csv(PROC_DIR / "calce_soh_per_cycle.csv")

    df["source"] = "calce"
    # Chemistry already in df (CS2 / CX2 = LiCoO2)
    if "chemistry" not in df.columns:
        df["chemistry"] = "LCO"

    if "avg_temp_c" not in df.columns:
        df["avg_temp_c"] = np.nan

    print(f"   ✅ {len(df):,} cycles | {df['cell_id'].nunique()} cells")
    return df


# ── Combine & harmonise ────────────────────────────────────────────────────
def combine(nasa, stanford, calce) -> pd.DataFrame:
    print("\n🔗 Combining datasets ...")

    # Keep only schema columns that exist in each df
    dfs = []
    for name, df in [("NASA", nasa), ("Stanford", stanford), ("CALCE", calce)]:
        available = [c for c in SCHEMA if c in df.columns]
        missing   = [c for c in SCHEMA if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = np.nan
        dfs.append(df[SCHEMA].copy())
        print(f"   {name:10s}: {len(df):6,} rows | missing cols filled: {missing}")

    combined = pd.concat(dfs, ignore_index=True)

    # Force correct types
    combined["cycle_number"]       = pd.to_numeric(combined["cycle_number"],       errors="coerce")
    combined["soh_pct"]            = pd.to_numeric(combined["soh_pct"],            errors="coerce")
    combined["cycle_capacity_ah"]  = pd.to_numeric(combined["cycle_capacity_ah"],  errors="coerce")
    combined["rul_cycles"]         = pd.to_numeric(combined["rul_cycles"],         errors="coerce").fillna(0).astype(int)
    combined["avg_voltage_v"]      = pd.to_numeric(combined["avg_voltage_v"],      errors="coerce")
    combined["avg_current_a"]      = pd.to_numeric(combined["avg_current_a"],      errors="coerce")
    combined["avg_temp_c"]         = pd.to_numeric(combined["avg_temp_c"],         errors="coerce")
    combined["internal_resistance"]= pd.to_numeric(combined["internal_resistance"],errors="coerce")

    print(f"\n   Total rows     : {len(combined):,}")
    print(f"   Total cells    : {combined['cell_id'].nunique()}")
    return combined


# ── Clean & validate ───────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧹 Cleaning & validating ...")
    before = len(df)

    # 1. Drop rows with missing SOH
    df = df.dropna(subset=["soh_pct"])
    print(f"   After dropping null SOH     : {len(df):,} rows")

    # 2. SOH must be in realistic range
    df = df[(df["soh_pct"] >= 5) & (df["soh_pct"] <= 110)]
    print(f"   After SOH range filter (5-110%): {len(df):,} rows")

    # 3. Capacity must be positive
    df = df[df["cycle_capacity_ah"] > 0]
    print(f"   After capacity > 0 filter   : {len(df):,} rows")

    # 4. Cycle number must be positive
    df = df[df["cycle_number"] > 0]
    print(f"   After cycle > 0 filter      : {len(df):,} rows")

    # 5. Re-compute status from SOH
    df["status"] = df["soh_pct"].apply(
        lambda s: "healthy" if s >= 80 else "end_of_life"
    )

    removed = before - len(df)
    print(f"\n   Removed {removed:,} invalid rows ({removed/before*100:.1f}%)")
    print(f"   Final clean dataset         : {len(df):,} rows")
    return df.reset_index(drop=True)


# ── Summary statistics ─────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("BATTERYIQ — COMBINED DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total cycles          : {len(df):,}")
    print(f"  Total cells           : {df['cell_id'].nunique()}")
    print(f"  SOH range             : {df['soh_pct'].min():.1f}% → {df['soh_pct'].max():.1f}%")
    print(f"  Healthy (≥80%)        : {(df['status']=='healthy').sum():,}")
    print(f"  EOL (<80%)            : {(df['status']=='end_of_life').sum():,}")
    print()
    print("  By source:")
    for src, grp in df.groupby("source"):
        print(f"    {src:12s}: {len(grp):7,} cycles | "
              f"{grp['cell_id'].nunique():3d} cells | "
              f"SOH {grp['soh_pct'].min():.0f}%→{grp['soh_pct'].max():.0f}%")
    print()
    print("  By chemistry:")
    for chem, grp in df.groupby("chemistry"):
        print(f"    {chem:8s}: {len(grp):7,} cycles | "
              f"{grp['cell_id'].nunique():3d} cells")
    print()
    print("  Missing values:")
    for col in ["avg_voltage_v","avg_current_a","avg_temp_c","internal_resistance"]:
        pct = df[col].isna().mean() * 100
        print(f"    {col:30s}: {pct:.1f}%")


# ── Plot combined SOH overview ─────────────────────────────────────────────
def plot_combined(df: pd.DataFrame):
    print("\n📈 Plotting combined SOH overview ...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sources   = ["nasa", "stanford", "calce"]
    colors    = plt.cm.tab20(np.linspace(0, 1, 20))
    titles    = ["NASA (34 cells, NMC)",
                 "Stanford (140 cells, LFP)",
                 "CALCE (15 cells, LCO)"]

    for ax, src, title in zip(axes, sources, titles):
        src_df = df[df["source"] == src]
        cells  = sorted(src_df["cell_id"].unique())
        # Plot max 30 cells for readability
        for i, cell in enumerate(cells[:30]):
            cdf = src_df[src_df["cell_id"] == cell].sort_values("cycle_number")
            ax.plot(cdf["cycle_number"], cdf["soh_pct"],
                    linewidth=0.8, alpha=0.5, color=colors[i % 20])
        ax.axhline(80, color="#EF4444", linestyle="--",
                   linewidth=2, label="EOL (80%)")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("SOH (%)")
        ax.set_ylim(0, 115)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("BatteryIQ — Combined Dataset: 189 Cells, 135K Cycles",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = PLOT_DIR / "combined_soh_overview.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Also save to memoire figures
    fig_mem = ROOT / "memoire" / "figures" / "fig10_combined_dataset_overview.png"
    import shutil
    shutil.copy(fig_path, fig_mem)
    print(f"   ✅ Saved → {fig_path.name}")
    print(f"   ✅ Saved → memoire/figures/fig10_combined_dataset_overview.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — Dataset Combination Pipeline")
    print("=" * 55)

    # 1. Load all sources
    nasa     = load_nasa()
    stanford = load_stanford()
    calce    = load_calce()

    # 2. Combine
    combined = combine(nasa, stanford, calce)

    # 3. Save raw combined
    raw_path = FEAT_DIR / "combined_soh.csv"
    combined.to_csv(raw_path, index=False)
    print(f"\n   💾 Raw combined → {raw_path.name} ({len(combined):,} rows)")

    # 4. Clean
    clean_df = clean(combined)

    # 5. Save clean version
    clean_path = FEAT_DIR / "combined_soh_clean.csv"
    clean_df.to_csv(clean_path, index=False)
    print(f"   💾 Clean combined → {clean_path.name} ({len(clean_df):,} rows)")

    # 6. Summary
    print_summary(clean_df)

    # 7. Plot
    plot_combined(clean_df)

    print("\n✅ Dataset combination complete!")
    print("   Next step: Feature Engineering →")
    print("   python pipeline/etl/05_feature_engineering.py")


if __name__ == "__main__":
    main()
