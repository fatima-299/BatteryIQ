"""
Check all sheets in CALCE Excel file
Run from BatteryIQ root: python check_calce3.py
"""
import pandas as pd
from pathlib import Path

RAW_DIR  = Path.cwd() / "data" / "raw" / "calce"
cell_dir = RAW_DIR / "CS2_33"
files    = sorted(cell_dir.glob("*.xlsx"))

# Check all sheets in first file
xl = pd.ExcelFile(files[0])
print(f"File: {files[0].name}")
print(f"Sheets: {xl.sheet_names}")

for sheet in xl.sheet_names:
    df = pd.read_excel(files[0], sheet_name=sheet, header=None, nrows=20)
    print(f"\n=== Sheet: '{sheet}' === shape before header: {df.shape}")
    print(df.head(8).to_string())

# Also check a larger file (more cycles)
print("\n\n=== Checking larger file ===")
# Find largest file
largest = max(files, key=lambda f: f.stat().st_size)
print(f"Largest file: {largest.name} ({largest.stat().st_size/1024:.0f} KB)")
xl2 = pd.ExcelFile(largest)
print(f"Sheets: {xl2.sheet_names}")
for sheet in xl2.sheet_names[:2]:
    df2 = pd.read_excel(largest, sheet_name=sheet, header=None, nrows=15)
    print(f"\nSheet '{sheet}': {df2.shape}")
    print(df2.head(10).to_string())
