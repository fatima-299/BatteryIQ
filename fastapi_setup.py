"""
Run this script from BatteryIQ root to create the full app structure.
  python fastapi_setup.py
"""
import os
from pathlib import Path

ROOT = Path(".")

# ── Folder structure ───────────────────────────────────────────────────────
folders = [
    "app/backend/api",
    "app/backend/services",
    "app/backend/schemas",
    "app/backend/models",
    "app/frontend/src/pages",
    "app/frontend/src/components",
    "app/frontend/src/hooks",
    "app/frontend/public",
]
for f in folders:
    Path(f).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created: {f}")

# ── requirements.txt for backend ──────────────────────────────────────────
backend_reqs = """fastapi==0.111.0
uvicorn==0.30.1
sqlalchemy==2.0.30
psycopg2-binary==2.9.9
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2
torch==2.3.1
xgboost==2.0.3
scikit-learn==1.5.0
openai==1.35.0
transformers==4.41.2
python-multipart==0.0.9
reportlab==4.2.2
Pillow==10.3.0
opencv-python-headless==4.10.0.82
python-dotenv==1.0.1
httpx==0.27.0
"""
Path("app/backend/requirements.txt").write_text(backend_reqs)
print("✅ Created: app/backend/requirements.txt")

# ── .env template ─────────────────────────────────────────────────────────
env_template = """# BatteryIQ Backend Environment Variables
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/batteryiq
MODEL_DIR=../../ml/models
FEAT_DIR=../../data/features
"""
Path("app/backend/.env").write_text(env_template)
print("✅ Created: app/backend/.env")

print("\n🎉 App structure ready!")
print("Next: pip install fastapi uvicorn sqlalchemy psycopg2-binary openai transformers reportlab opencv-python-headless python-dotenv")
