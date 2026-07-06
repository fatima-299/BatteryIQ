"""
Quick check of Stanford .mat file structure
Run from BatteryIQ root:
    python check_stanford.py
"""
import h5py
import numpy as np
from pathlib import Path

RAW_DIR = Path.cwd() / "data" / "raw" / "stanford"
mat_files = sorted(RAW_DIR.glob("*.mat"))

print(f"Found {len(mat_files)} .mat files:")
for f in mat_files:
    print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.0f} MB)")

# Inspect first file only
f = mat_files[0]
print(f"\n🔍 Inspecting: {f.name}")

with h5py.File(f, "r") as h:
    print(f"\nTop-level keys: {list(h.keys())}")

    # Go one level deeper
    for key in list(h.keys())[:3]:
        print(f"\n  [{key}] type: {type(h[key])}")
        if hasattr(h[key], "keys"):
            sub = list(h[key].keys())[:5]
            print(f"    Sub-keys (first 5): {sub}")
            for sk in sub[:2]:
                print(f"      [{sk}] → {type(h[key][sk])}")
                if hasattr(h[key][sk], "keys"):
                    print(f"        keys: {list(h[key][sk].keys())[:5]}")
