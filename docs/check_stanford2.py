"""
Deep inspection of Stanford batch structure
Run from BatteryIQ root:
    python check_stanford2.py
"""
import h5py
import numpy as np
from pathlib import Path

RAW_DIR  = Path.cwd() / "data" / "raw" / "stanford"
mat_file = sorted(RAW_DIR.glob("*.mat"))[0]

print(f"🔍 Deep inspection: {mat_file.name}\n")

with h5py.File(mat_file, "r") as h:
    batch = h["batch"]
    print(f"batch keys: {list(batch.keys())}")

    # cycle_life — how many cycles each cell lived
    cycle_life = batch["cycle_life"][:]
    print(f"\ncycle_life shape : {cycle_life.shape}")
    print(f"cycle_life values: {cycle_life.flatten()[:10]} ...")

    # summary — cycle-level capacity data
    print(f"\nbatch/summary keys: {list(batch['summary'].keys())}")

    # QDischarge — the capacity per cycle per cell
    Qd = batch["summary"]["QDischarge"]
    print(f"\nQDischarge type  : {type(Qd)}")
    print(f"QDischarge shape : {Qd.shape if hasattr(Qd,'shape') else 'ref-based'}")

    # Check if it's references
    if Qd.dtype == object or str(Qd.dtype) == "|O":
        print("→ QDischarge contains HDF5 references (one per cell)")
        # Dereference first cell
        ref = Qd[0, 0]
        cell_cap = h[ref][:]
        print(f"  Cell 0 capacity shape: {cell_cap.shape}")
        print(f"  Cell 0 first 5 values: {cell_cap.flatten()[:5]}")
    else:
        print(f"→ Direct array, first values: {Qd[:5]}")

    # Check cycles structure
    print(f"\nbatch/cycles keys (first 5): {list(batch['cycles'].keys())[:5]}")
    cyc0_key = list(batch['cycles'].keys())[0]
    cyc0 = batch['cycles'][cyc0_key]
    print(f"cycles[{cyc0_key}] keys: {list(cyc0.keys())}")

    # Tavg — temperature per cycle
    if "Tavg" in batch["summary"]:
        Tavg = batch["summary"]["Tavg"]
        print(f"\nTavg shape: {Tavg.shape}")

    # IR — internal resistance
    if "IR" in batch["summary"]:
        IR = batch["summary"]["IR"]
        print(f"IR shape: {IR.shape}")
