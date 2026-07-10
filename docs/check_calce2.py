"""
Deep check of CALCE Excel structure
Run from BatteryIQ root: python check_calce2.py
"""
import pandas as pd
from pathlib import Path

RAW_DIR  = Path.cwd() / "data" / "raw" / "calce"
cell_dir = RAW_DIR / "CS2_33"
files    = sorted(cell_dir.glob("*.xlsx"))

print(f"First file: {files[0].name}\n")

# Read first 15 rows raw — no header assumption
df_raw = pd.read_excel(files[0], header=None, nrows=15)
print("=== First 15 rows (raw, no header) ===")
print(df_raw.to_string())

print("\n=== Number of columns ===")
print(df_raw.shape[1])

# Also check the last file (may have different structure)
print(f"\nLast file: {files[-1].name}")
df_last = pd.read_excel(files[-1], header=None, nrows=10)
print(df_last.to_string())
