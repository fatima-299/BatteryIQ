"""
BatteryIQ — NASA Battery Dataset Ingestion (v2 - fixed SOH)
=============================================================
Run from BatteryIQ root:
    python pipeline/ingestion/01_nasa_ingestion.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
RAW_DIR   = ROOT / "data" / "raw" / "nasa"
DATA_DIR  = RAW_DIR / "data"
META_PATH = RAW_DIR / "metadata.csv"
PROC_DIR  = ROOT / "data" / "processed"
PLOT_DIR  = PROC_DIR / "plots"
PROC_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Load metadata ──────────────────────────────────────────────────
def load_metadata() -> pd.DataFrame:
    print("📋 Loading metadata.csv ...")
    meta = pd.read_csv(META_PATH)
    # Force Capacity to numeric — anything non-numeric becomes NaN
    meta["Capacity"] = pd.to_numeric(meta["Capacity"], errors="coerce")
    print(f"   Total entries     : {len(meta)}")
    print(f"   Cycle types       : {meta['type'].value_counts().to_dict()}")
    print(f"   Unique batteries  : {meta['battery_id'].nunique()}")
    return meta


# ── Step 2: Load one measurement CSV ──────────────────────────────────────
def load_measurement(filename: str) -> pd.DataFrame:
    path = DATA_DIR / str(filename).strip()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# ── Step 3: Merge metadata + measurements ─────────────────────────────────
def build_full_dataset(meta: pd.DataFrame) -> pd.DataFrame:
    print("\n🔗 Merging metadata with measurement files ...")
    discharge_meta = meta[
        (meta["type"] == "discharge") & (meta["Capacity"].notna())
    ].copy().reset_index(drop=True)
    print(f"   Discharge cycles with valid Capacity: {len(discharge_meta)}")

    all_rows = []
    for idx, row in discharge_meta.iterrows():
        df_meas = load_measurement(row["filename"])
        if df_meas.empty:
            continue
        df_meas["battery_id"]        = row["battery_id"]
        df_meas["test_id"]           = row["test_id"]
        df_meas["uid"]               = row["uid"]
        df_meas["filename"]          = str(row["filename"]).strip()
        df_meas["cycle_type"]        = row["type"]
        df_meas["ambient_temp_c"]    = pd.to_numeric(row["ambient_temperature"], errors="coerce")
        df_meas["cycle_capacity_ah"] = float(row["Capacity"])
        df_meas["Re"]                = pd.to_numeric(row.get("Re",  np.nan), errors="coerce")
        df_meas["Rct"]               = pd.to_numeric(row.get("Rct", np.nan), errors="coerce")
        all_rows.append(df_meas)
        if (idx + 1) % 200 == 0:
            print(f"   Processed {idx+1}/{len(discharge_meta)} files ...")

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.rename(columns={
        "Voltage_measured"    : "voltage_v",
        "Current_measured"    : "current_a",
        "Temperature_measured": "temp_c",
        "Current_load"        : "current_load_a",
        "Voltage_load"        : "voltage_load_v",
        "Time"                : "time_s",
    }, inplace=True)
    print(f"\n   ✅ Full dataset shape : {full_df.shape}")
    return full_df


# ── Step 4: Compute SOH per cycle ─────────────────────────────────────────
def compute_soh(full_df: pd.DataFrame) -> pd.DataFrame:
    print("\n📊 Computing SOH per cycle ...")

    soh_df = (
        full_df.groupby(["battery_id", "test_id", "uid", "filename"])
        .agg(
            cycle_capacity_ah=("cycle_capacity_ah", "first"),
            ambient_temp_c   =("ambient_temp_c",    "first"),
            avg_voltage_v    =("voltage_v",          "mean"),
            avg_current_a    =("current_a",          "mean"),
            avg_temp_c       =("temp_c",             "mean"),
            Re               =("Re",                 "first"),
            Rct              =("Rct",                "first"),
            n_measurements   =("voltage_v",          "count"),
        )
        .reset_index()
        .sort_values(["battery_id", "test_id"])
    )

    soh_rows = []
    skipped  = []

    for battery_id, group in soh_df.groupby("battery_id"):
        group = group.copy().reset_index(drop=True)

        caps = group["cycle_capacity_ah"].dropna()
        if len(caps) < 3:
            skipped.append(battery_id)
            continue

        # ── KEY FIX: nominal = median of top-10% capacity values per battery
        # This handles batteries with different cell sizes correctly
        top_10pct   = caps.quantile(0.90)
        nominal_cap = caps[caps <= top_10pct * 1.05].median()

        if nominal_cap <= 0 or np.isnan(nominal_cap):
            skipped.append(battery_id)
            continue

        group["nominal_capacity_ah"] = round(nominal_cap, 4)
        group["soh_pct"]             = (group["cycle_capacity_ah"] / nominal_cap * 100).round(2)
        group["cycle_number"]        = range(1, len(group) + 1)

        # Cap SOH at 105% (anything above is a sensor artefact)
        group = group[group["soh_pct"] <= 105].copy()
        group["cycle_number"] = range(1, len(group) + 1)

        # RUL: cycles remaining before SOH < 80%
        eol_cycles  = group[group["soh_pct"] < 80]
        eol_at      = eol_cycles["cycle_number"].min() if not eol_cycles.empty else None
        group["rul_cycles"] = group["cycle_number"].apply(
            lambda c: int(eol_at - c) if eol_at and c < eol_at else 0
        )
        group["status"] = group["soh_pct"].apply(
            lambda s: "healthy" if s >= 80 else "end_of_life"
        )

        print(f"   {battery_id}: {len(group):3d} cycles | "
              f"nominal={nominal_cap:.4f}Ah | "
              f"SOH {group['soh_pct'].max():.1f}% → {group['soh_pct'].min():.1f}%")
        soh_rows.append(group)

    if skipped:
        print(f"\n   ⚠️  Skipped batteries (insufficient data): {skipped}")

    return pd.concat(soh_rows, ignore_index=True)


# ── Step 5: Plot SOH curves ────────────────────────────────────────────────
def plot_soh_curves(soh_df: pd.DataFrame):
    print("\n📈 Generating SOH plots ...")
    batteries = sorted(soh_df["battery_id"].unique())
    colors    = ["#378ADD","#EF9F27","#7F77DD","#1D9E75",
                 "#EF4444","#F97316","#8B5CF6","#06B6D4",
                 "#EC4899","#14B8A6","#F59E0B","#6366F1"]

    # Individual plots
    for battery_id in batteries:
        df = soh_df[soh_df["battery_id"] == battery_id]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["cycle_number"], df["soh_pct"],
                color="#378ADD", linewidth=1.5, label=battery_id)
        ax.axhline(80, color="#EF4444", linestyle="--",
                   linewidth=1, label="EOL threshold (80%)")
        ax.fill_between(df["cycle_number"], df["soh_pct"], 80,
                        where=df["soh_pct"] < 80,
                        alpha=0.15, color="#EF4444")
        ax.set_title(f"BatteryIQ — {battery_id} State of Health", fontsize=13)
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("SOH (%)")
        ax.set_ylim(50, 108)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{battery_id}_soh.png", dpi=150)
        plt.close()

    # Combined overview
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, battery_id in enumerate(batteries):
        df = soh_df[soh_df["battery_id"] == battery_id]
        ax.plot(df["cycle_number"], df["soh_pct"],
                linewidth=1.0, label=battery_id,
                color=colors[i % len(colors)], alpha=0.8)
    ax.axhline(80, color="#EF4444", linestyle="--",
               linewidth=2, label="EOL threshold (80%)")
    ax.set_title("BatteryIQ — All Batteries SOH Overview (NASA Dataset)", fontsize=13)
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("SOH (%)")
    ax.set_ylim(50, 108)
    ax.legend(loc="lower left", fontsize=7, ncol=5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "all_batteries_soh_overview.png", dpi=150)
    plt.close()
    print(f"   ✅ {len(batteries)} battery plots + 1 overview saved → {PLOT_DIR}")


# ── Step 6: Quality report ────────────────────────────────────────────────
def quality_report(full_df: pd.DataFrame, soh_df: pd.DataFrame):
    print("\n🔍 Data Quality Report")
    print("=" * 45)
    print(f"  Total measurements     : {len(full_df):,}")
    print(f"  Total discharge cycles : {len(soh_df):,}")
    print(f"  Unique batteries       : {soh_df['battery_id'].nunique()}")
    print(f"  Missing values:")
    for col in ["voltage_v", "current_a", "temp_c", "cycle_capacity_ah"]:
        if col in full_df.columns:
            n = full_df[col].isna().sum()
            print(f"    {col:30s}: {n} nulls")
    print(f"  SOH range              : {soh_df['soh_pct'].min():.1f}% → {soh_df['soh_pct'].max():.1f}%")
    print(f"  Healthy cycles (≥80%)  : {(soh_df['status']=='healthy').sum()}")
    print(f"  EOL cycles (<80%)      : {(soh_df['status']=='end_of_life').sum()}")
    print(f"  Batteries reaching EOL : {soh_df[soh_df['status']=='end_of_life']['battery_id'].nunique()}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — NASA Ingestion Pipeline v2")
    print("=" * 45)
    meta    = load_metadata()
    full_df = build_full_dataset(meta)
    soh_df  = compute_soh(full_df)
    plot_soh_curves(soh_df)

    print("\n💾 Saving processed files ...")
    full_df.to_csv(PROC_DIR / "nasa_all_measurements.csv", index=False)
    soh_df.to_csv(PROC_DIR  / "nasa_soh_per_cycle.csv",   index=False)
    print(f"   ✅ nasa_all_measurements.csv  ({len(full_df):,} rows)")
    print(f"   ✅ nasa_soh_per_cycle.csv     ({len(soh_df):,} rows)")

    quality_report(full_df, soh_df)
    print("\n✅ NASA ingestion complete — data is clean and ready for EDA!")

if __name__ == "__main__":
    main()
