# BatteryIQ — Physics-Informed ML for EV Battery Degradation Prediction

BatteryIQ is a complete fleet intelligence system for predicting electric vehicle battery
degradation. It combines a Physics-Informed Neural Network (PINN) with dual physical
constraints (SEI growth + Arrhenius thermal degradation) with an end-to-end production
pipeline: distributed ETL, a PostgreSQL warehouse, a live Power BI dashboard, and a web
application with four AI modules.

Master Data Engineering & Big Data — Sorbonne Data Analytics.

Full write-up, methodology, and results: see the thesis (`memoire/`) or the [full report](#)
[← add your GitHub Pages / PDF link here if you have one].

---

## 1. What's in this repository

```
BatteryIQ/
├── app/                  ← FastAPI backend + React frontend (the running application)
│   ├── backend/          ← FastAPI app: main.py + 6 API routers
│   └── frontend/         ← React app: 5 pages, 4 AI modules
├── dashboard/            ← Power BI dashboard (.pbix file)
├── data/                 ← raw/, processed/, features/ (see Datasets below — not all committed)
├── docs/                 ← architecture notes, ad-hoc data-check scripts
├── memoire/              ← the mémoire itself + all 41 figures used in it
│   └── figures/          ← publication-ready figures (fig01–fig37 + variants)
├── ml/                   ← model training, evaluation, explainability, validation
├── notebooks/            ← EDA Jupyter notebooks (02–05)
├── pipeline/             ← ingestion scripts (01–03) + ETL (04–06, PySpark)
├── scripts/              ← one-off setup/scaffolding scripts (not part of the running app)
├── tests/                ← unit/integration tests
├── env.example           ← template for your own .env file
├── requirements.txt      ← root-level Python dependencies
├── run.sh                ← starts backend + frontend together, one command
└── README.md             ← this file
```

**Datasets used:** NASA PCoE (34 NMC cells, 18650 format), Stanford/MIT MATR (140 LFP cells),
CALCE/University of Maryland (15 LiCoO₂ cells, CS2 + CX2), and RWTH Aachen (48 NMC cells,
real drive-cycle profiles — downloaded but not yet processed, see thesis Recommendations).
Combined: **134,938 cycles across 189 cells and 4 chemistries.**

---

## 2. Prerequisites

- **Python 3.14** (the backend was built and tested specifically against this version)
- **Node.js** + npm (for the React frontend)
- **PostgreSQL 16** (or compatible)
- **Apache Spark 4.2** (only needed if you're re-running the ETL pipeline from raw data)
- An **OpenAI API key** (only needed for the BatteryChat module)
- Power BI Desktop (only needed to open/edit `dashboard/*.pbix`)

---

## 3. First-time setup

### 3.1 Clone and install dependencies

```bash
git clone https://github.com/fatima-299/BatteryIQ.git
cd BatteryIQ

# Backend
cd app/backend
pip install -r requirements.txt
cd ../..

# Frontend
cd app/frontend
npm install
cd ../..
```

### 3.2 Set up PostgreSQL

Create the database and load the schema (matches exactly what the ETL pipeline writes and
what the backend/dashboard both query):

```bash
psql -U postgres -c "CREATE DATABASE batteryiq;"
psql -U postgres -d batteryiq -f pipeline/schema.sql
```

If you're not running the full ETL pipeline yourself, you'll need to load your own processed
data into the `battery_cycles` table before the dashboard or predictions will show real
results — an empty table means an empty dashboard, not an error.

### 3.3 Configure environment variables

```bash
cp env.example app/backend/.env
```

Edit `app/backend/.env` and fill in:

```
OPENAI_API_KEY=sk-...your real key...
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/batteryiq
MODEL_DIR=../../ml/models
FEAT_DIR=../../data/features
```

`app/backend/.env` is git-ignored — never commit your real API key.

---

## 4. Running the application

### Option A — one command (recommended)

From the repository root:

```bash
chmod +x run.sh
./run.sh
```

This starts both the backend and frontend together, and stops both cleanly on Ctrl+C.

### Option B — two terminals

**Terminal 1 (backend):**
```bash
cd app/backend
python -m uvicorn main:app --reload --port 8000 --host 0.0.0.0
```

**Terminal 2 (frontend):**
```bash
cd app/frontend
npm start
```

> **Windows users:** always include `--host 0.0.0.0` on the uvicorn command. Without it, the
> backend only listens on `127.0.0.1`, but Windows often resolves `localhost` (which the
> frontend calls) to the IPv6 address `::1` — causing "Cannot connect to API" errors even
> though the backend is running fine. `run.sh` already includes this flag.

Once both are running:
- Frontend: **http://localhost:3000**
- Backend + interactive API docs: **http://127.0.0.1:8000/docs**

---

## 5. Using the application

| Page | What it does |
|---|---|
| **Fleet Dashboard** | All monitored cells in one table — filter by alert status, search, sort by risk score |
| **Cell Deep-Dive** | Full SOH degradation trajectory for any single cell, by cell ID |
| **BatteryChat** | GPT-4o assistant with RAG — select a cell, ask about its health; answers are grounded in that cell's real data, not generic knowledge |
| **Image Analyser** | Upload a battery photo — OpenCV-based heuristic defect detection (swelling, corrosion, SEI buildup) |
| **Report Generator** | Paste a maintenance log for NLP analysis, or generate a full GPT-4o-written PDF health report for any cell |

The **Power BI dashboard** (`dashboard/*.pbix`) connects live via DirectQuery to the same
PostgreSQL database — open it in Power BI Desktop and point it at your local `batteryiq`
database to see the same data from a BI perspective (4 pages: Fleet Overview, Cell Deep-Dive,
Predictive Alerts, Physics Analytics).

---

## 6. Re-running the data pipeline (optional)

Only needed if you want to reprocess data from scratch rather than using an existing database.

```bash
# 1. Ingest each source
python pipeline/ingestion/01_nasa_ingestion.py
python pipeline/ingestion/02_stanford_ingestion.py
python pipeline/ingestion/03_calce_ingestion.py

# 2. Combine + engineer features
python pipeline/etl/04_combine_sources.py
python pipeline/etl/05_feature_engineering.py

# 3. Run the PySpark ETL (writes to PostgreSQL)
python pipeline/etl/06_pyspark_etl.py
```

Model training scripts live in `ml/training/`; trained artefacts (`.pkl`, `.pt`) are written
to `ml/models/` and loaded automatically by the backend at startup.

---

## 7. Known limitations (documented honestly, see thesis for full detail)

- **NASA NMC data cannot train the PINN**: 100% missing internal resistance, near-zero
  temperature variation, and only ~55 cycles per cell on average. XGBoost still handles it
  reasonably (RMSE≈2.16%), but physics constraints have nothing to act on.
- **Image Analyser** uses classical OpenCV heuristics, not a trained CNN — it can misread
  dark backgrounds as corrosion. A labelled-image CNN classifier is a natural next step.
- **HNEI cross-dataset validation** isn't possible without a feature-schema adapter — its
  pre-engineered features are structurally different from this project's schema.

---

## 8. Tech stack

| Layer | Technology |
|---|---|
| Distributed ETL | Apache Spark 4.2 |
| Data warehouse | PostgreSQL 16 |
| ML — baseline | XGBoost 2.0 |
| ML — sequence + physics-informed | PyTorch 2.3 (LSTM, PINN) |
| Backend API | FastAPI 0.111, Python 3.14 |
| Frontend | React 18 |
| LLM module | GPT-4o (OpenAI), RAG architecture |
| BI dashboard | Power BI Desktop, DirectQuery |

---

## 9. License

[Add your license here — e.g. MIT, or "Academic project, all rights reserved" if this is
submitted coursework not intended for reuse.]
