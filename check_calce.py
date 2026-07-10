"""
Check CALCE Excel file structure
Run from BatteryIQ root: python check_calce.py
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path.cwd() / "data" / "raw" / "calce"
cell_dir = RAW_DIR / "CS2_33"
files    = sorted(cell_dir.glob("*.xlsx"))

print(f"Files in CS2_33: {len(files)}")
print(f"First file: {files[0].name}")

# Read first file
df = pd.read_excel(files[0])
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nData types:")
print(df.dtypes)
