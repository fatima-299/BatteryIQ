"""Check CX2_4 folder contents"""
from pathlib import Path
RAW_DIR = Path.cwd() / "data" / "raw" / "calce" / "CX2_4"
print(f"Folder exists: {RAW_DIR.exists()}")
if RAW_DIR.exists():
    files = list(RAW_DIR.iterdir())
    print(f"Total files: {len(files)}")
    for f in files[:10]:
        print(f"  {f.name} ({f.suffix})")
