"""
Final structure check — dereference HDF5 refs
Run from BatteryIQ root: python check_stanford3.py
"""
import h5py
import numpy as np
from pathlib import Path

RAW_DIR  = Path.cwd() / "data" / "raw" / "stanford"
mat_file = sorted(RAW_DIR.glob("*.mat"))[0]

print(f"🔍 {mat_file.name}\n")

with h5py.File(mat_file, "r") as h:
    batch = h["batch"]

    # 1. How many cells in this batch?
    n_cells = batch["cycle_life"].shape[0]
    print(f"Number of cells: {n_cells}")

    # 2. Dereference cycle_life for first cell
    ref0 = batch["cycle_life"][0, 0]
    life0 = h[ref0][0, 0]
    print(f"Cell 0 cycle life: {int(life0)} cycles")

    # 3. What is summary? — it's a Group with ref arrays
    summary = batch["summary"]
    print(f"\nsummary type: {type(summary)}")

    # Try treating summary as a dataset of references
    sum_data = summary[()]
    print(f"summary shape: {sum_data.shape}")
    print(f"summary dtype: {sum_data.dtype}")

    # Dereference summary[0] for cell 0
    ref_sum0 = sum_data[0, 0]
    cell_sum = h[ref_sum0]
    print(f"\nCell 0 summary type: {type(cell_sum)}")
    if hasattr(cell_sum, 'keys'):
        print(f"Cell 0 summary keys: {list(cell_sum.keys())}")
        # Get QDischarge
        if "QDischarge" in cell_sum:
            Qd = cell_sum["QDischarge"][:]
            print(f"QDischarge shape: {Qd.shape}")
            print(f"QDischarge first 5: {Qd.flatten()[:5]}")
        if "Tavg" in cell_sum:
            T = cell_sum["Tavg"][:]
            print(f"Tavg first 5: {T.flatten()[:5]}")
        if "IR" in cell_sum:
            IR = cell_sum["IR"][:]
            print(f"IR first 5: {IR.flatten()[:5]}")
    else:
        # Summary is a flat array
        arr = cell_sum[:]
        print(f"Summary array shape: {arr.shape}, values: {arr.flatten()[:5]}")

    # 4. Check cycles for cell 0
    cycles_data = batch["cycles"]
    print(f"\ncycles type: {type(cycles_data)}")
    cyc_arr = cycles_data[()]
    print(f"cycles shape: {cyc_arr.shape}")
    ref_cyc0 = cyc_arr[0, 0]
    cell_cyc = h[ref_cyc0]
    print(f"Cell 0 cycles type: {type(cell_cyc)}")
    if hasattr(cell_cyc, 'keys'):
        print(f"Cell 0 cycles keys: {list(cell_cyc.keys())}")
