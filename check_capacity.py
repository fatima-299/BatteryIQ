"""
Quick diagnostic — run from BatteryIQ root:
    python check_capacity.py
"""
import pandas as pd
from pathlib import Path

ROOT     = Path.cwd()
META     = ROOT / "data" / "raw" / "nasa" / "metadata.csv"
SOH_CSV  = ROOT / "data" / "processed" / "nasa_soh_per_cycle.csv"

meta = pd.read_csv(META)
soh  = pd.read_csv(SOH_CSV)

print("=== Capacity column in metadata ===")
print(meta["Capacity"].dtype)
print(meta["Capacity"].head(10).to_string())

print("\n=== Discharge rows with valid Capacity ===")
discharge = meta[meta["type"] == "discharge"]
print(f"Total discharge rows : {len(discharge)}")
print(f"Non-null Capacity    : {discharge['Capacity'].notna().sum()}")
print(f"Numeric-convertible  : {pd.to_numeric(discharge['Capacity'], errors='coerce').notna().sum()}")

print("\n=== SOH stats per battery (top 10) ===")
stats = soh.groupby("battery_id")["soh_pct"].agg(["min","max","count"])
print(stats.head(10).to_string())

print("\n=== Rows where SOH > 200% (corrupted) ===")
bad = soh[soh["soh_pct"] > 200]
print(f"Count: {len(bad)}")
print(bad[["battery_id","cycle_number","cycle_capacity_ah","soh_pct"]].head(10).to_string())

print("\n=== Valid SOH range (0-120%) ===")
valid = soh[(soh["soh_pct"] > 0) & (soh["soh_pct"] <= 120)]
print(f"Valid cycles: {len(valid)} / {len(soh)}")
print(f"Valid SOH range: {valid['soh_pct'].min():.1f}% → {valid['soh_pct'].max():.1f}%")
