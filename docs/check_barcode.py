"""
Check how barcodes are stored in Stanford HDF5
Run from BatteryIQ root: python check_barcode.py
"""
import h5py
import numpy as np
from pathlib import Path

RAW_DIR  = Path.cwd() / "data" / "raw" / "stanford"
mat_file = sorted(RAW_DIR.glob("*.mat"))[0]

with h5py.File(mat_file, "r") as h:
    batch = h["batch"]

    print("barcode dataset:")
    bc = batch["barcode"]
    print(f"  shape : {bc.shape}")
    print(f"  dtype : {bc.dtype}")

    # Try first 3 cells
    for i in range(3):
        ref = bc[i, 0]
        raw = h[ref][()]
        print(f"\n  cell {i} raw type : {type(raw)}, dtype: {raw.dtype}, shape: {raw.shape}")
        print(f"  cell {i} raw values: {raw.flatten()[:10]}")
        # Try decode as uint16 chars
        try:
            s = "".join(chr(c) for c in raw.flatten())
            print(f"  cell {i} as string: '{s}'")
        except Exception as e:
            print(f"  decode error: {e}")
