"""
BatteryIQ — CALCE Battery Dataset Ingestion v3
================================================
Fixed: Discharge_Capacity is cumulative — use max-min per cycle

Run from BatteryIQ root:
  python pipeline/ingestion/03_calce_ingestion.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[2]
RAW_DIR  = ROOT / "data" / "raw" / "calce"
PROC_DIR = ROOT / "data" / "processed"
PLOT_DIR = PROC_DIR / "plots"
PROC_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CELLS = [
    "CS2_3","CS2_9","CS2_33","CS2_34","CS2_35","CS2_36","CS2_37","CS2_38",
    "CX2_16","CX2_33","CX2_34","CX2_35","CX2_36","CX2_37","CX2_38",
]

NOMINAL = {"CS2": 1.1, "CX2": 1.35}


def read_excel_file(filepath: Path) -> pd.DataFrame:
    try:
        engine = "xlrd" if filepath.suffix == ".xls" else "openpyxl"
        xl     = pd.ExcelFile(filepath, engine=engine)
        sheets = [s for s in xl.sheet_names if s != "Info"]
        if not sheets:
            return pd.DataFrame()
        df = pd.read_excel(filepath, sheet_name=sheets[0],
                           header=0, engine=engine)
        df.columns = df.columns.str.strip()
        if "Cycle_Index" not in df.columns:
            return pd.DataFrame()
        df = df[pd.to_numeric(df["Cycle_Index"], errors="coerce").notna()].copy()
        df["Cycle_Index"] = df["Cycle_Index"].astype(int)
        return df
    except Exception:
        return pd.DataFrame()


def process_cell(cell_name: str) -> pd.DataFrame:
    cell_dir  = RAW_DIR / cell_name
    if not cell_dir.exists():
        return pd.DataFrame()
    xlsx_files = sorted(list(cell_dir.glob("*.xlsx")) +
                        list(cell_dir.glob("*.xls")))
    if not xlsx_files:
        return pd.DataFrame()

    all_dfs      = []
    cycle_offset = 0

    for f in xlsx_files:
        df = read_excel_file(f)
        if df.empty:
            continue
        max_cycle     = df["Cycle_Index"].max()
        df["Cycle_Index"] = df["Cycle_Index"] + cycle_offset
        cycle_offset += max_cycle
        df["file"]    = f.stem
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    rename = {
        "Cycle_Index"              : "cycle_index",
        "Current(A)"               : "current_a",
        "Voltage(V)"               : "voltage_v",
        "Discharge_Capacity(Ah)"   : "discharge_cap_ah",
        "Charge_Capacity(Ah)"      : "charge_cap_ah",
        "Internal_Resistance(Ohm)" : "internal_resistance",
        "dV/dt(V/s)"               : "dvdt",
        "Test_Time(s)"             : "test_time_s",
        "Date_Time"                : "datetime",
    }
    combined.rename(columns={k: v for k, v in rename.items()
                             if k in combined.columns}, inplace=True)

    combined["cell_id"]   = cell_name
    combined["source"]    = "calce"
    combined["chemistry"] = "CS2" if cell_name.startswith("CS2") else "CX2"

    for col in ["cycle_index","current_a","voltage_v",
                "discharge_cap_ah","internal_resistance"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return combined


def compute_soh(raw_df: pd.DataFrame) -> pd.DataFrame:
    soh_rows = []

    for cell_id, cell_df in raw_df.groupby("cell_id"):
        chemistry = cell_df["chemistry"].iloc[0]
        nominal   = NOMINAL.get(chemistry, 1.1)

        for cycle_num, cyc in cell_df.groupby("cycle_index"):
            if cycle_num < 1:
                continue

            # ── KEY FIX: actual capacity = max - min of cumulative column ──
            cap_series = cyc["discharge_cap_ah"].dropna() \
                         if "discharge_cap_ah" in cyc.columns \
                         else pd.Series(dtype=float)

            if len(cap_series) < 2:
                continue

            cap = float(cap_series.max() - cap_series.min())

            if cap <= 0.05:   # filter near-zero (charge cycles, rest steps)
                continue

            soh = round(cap / nominal * 100, 2)
            if soh > 110:
                continue

            soh_rows.append({
                "cell_id"            : cell_id,
                "source"             : "calce",
                "chemistry"          : chemistry,
                "cycle_number"       : int(cycle_num),
                "cycle_capacity_ah"  : round(cap, 6),
                "nominal_capacity_ah": nominal,
                "soh_pct"            : soh,
                "avg_voltage_v"      : round(cyc["voltage_v"].mean(), 4)
                                       if "voltage_v" in cyc.columns else np.nan,
                "avg_current_a"      : round(cyc["current_a"].mean(), 4)
                                       if "current_a" in cyc.columns else np.nan,
                "internal_resistance": round(cyc["internal_resistance"].mean(), 6)
                                       if "internal_resistance" in cyc.columns
                                       else np.nan,
                "status"             : "healthy" if soh >= 80 else "end_of_life",
            })

    soh_df = pd.DataFrame(soh_rows).sort_values(["cell_id","cycle_number"])

    rul_chunks = []
    for cell_id, grp in soh_df.groupby("cell_id"):
        grp = grp.copy().reset_index(drop=True)
        eol = grp[grp["soh_pct"] < 80]["cycle_number"].min()
        grp["rul_cycles"] = grp["cycle_number"].apply(
            lambda c: int(eol - c) if pd.notna(eol) and c < eol else 0
        )
        rul_chunks.append(grp)

    return pd.concat(rul_chunks, ignore_index=True)


def plot_soh(soh_df: pd.DataFrame):
    print("\n📈 Plotting SOH curves ...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors    = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, chemistry in zip(axes, ["CS2", "CX2"]):
        chem_df = soh_df[soh_df["chemistry"] == chemistry]
        for i, cell in enumerate(sorted(chem_df["cell_id"].unique())):
            df = chem_df[chem_df["cell_id"] == cell].sort_values("cycle_number")
            ax.plot(df["cycle_number"], df["soh_pct"],
                    linewidth=1.2, alpha=0.8,
                    color=colors[i % 10], label=cell)
        ax.axhline(80, color="#EF4444", linestyle="--",
                   linewidth=1.5, label="EOL 80%")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("SOH (%)")
        ax.set_ylim(50, 112)
        ax.set_title(f"CALCE {chemistry} — SOH Degradation", fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle("BatteryIQ — CALCE Dataset SOH Overview",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "calce_soh_overview.png", dpi=150)
    plt.close()
    print("   ✅ Plot saved → calce_soh_overview.png")


def main():
    print("\n🔋 BatteryIQ — CALCE Ingestion v3 (cumulative cap fix)")
    print("=" * 55)

    all_raw = []
    for cell_name in CELLS:
        print(f"   Processing {cell_name} ...", end=" ")
        df = process_cell(cell_name)
        if df.empty:
            print("❌ empty")
            continue
        print(f"✅ {len(df):,} rows, {df['cycle_index'].nunique()} cycles")
        all_raw.append(df)

    if not all_raw:
        print("⚠️  No data extracted!")
        return

    raw_df = pd.concat(all_raw, ignore_index=True)

    print("\n📊 Computing SOH per cycle ...")
    soh_df = compute_soh(raw_df)

    plot_soh(soh_df)

    soh_path = PROC_DIR / "calce_soh_per_cycle.csv"
    soh_df.to_csv(soh_path, index=False)

    print("\n🔍 CALCE Data Quality Report")
    print("=" * 55)
    print(f"  Total cycles       : {len(soh_df):,}")
    print(f"  Unique cells       : {soh_df['cell_id'].nunique()}")
    print(f"  CS2 cells          : {soh_df[soh_df['chemistry']=='CS2']['cell_id'].nunique()}")
    print(f"  CX2 cells          : {soh_df[soh_df['chemistry']=='CX2']['cell_id'].nunique()}")
    print(f"  SOH range          : {soh_df['soh_pct'].min():.1f}% → {soh_df['soh_pct'].max():.1f}%")
    print(f"  Healthy (≥80%)     : {(soh_df['status']=='healthy').sum():,}")
    print(f"  EOL (<80%)         : {(soh_df['status']=='end_of_life').sum():,}")
    print(f"  Missing IR values  : {soh_df['internal_resistance'].isna().sum()}")
    print(f"\n  Per cell cycles:")
    for cell, grp in soh_df.groupby("cell_id"):
        print(f"    {cell:12s}: {len(grp):5d} cycles | "
              f"SOH {grp['soh_pct'].max():.0f}%→{grp['soh_pct'].min():.0f}%")
    print(f"\n  ✅ Saved → {soh_path}")
    print("\n✅ CALCE ingestion complete!")


if __name__ == "__main__":
    main()
