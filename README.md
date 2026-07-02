# BatteryIQ 🔋
**Physics-Informed Machine Learning for EV Battery Degradation Prediction**

Master Mémoire — Data Engineering & Big Data

## Project Structure
- `data/` — raw datasets (NASA, CALCE, Stanford, RWTH) + processed + features
- `pipeline/` — ingestion scripts, PySpark ETL, data validation
- `ml/` — model training, evaluation, SHAP explainability
- `app/` — FastAPI backend + React frontend
- `dashboard/` — Power BI files (.pbix)
- `memoire/` — chapters, figures, references
- `notebooks/` — EDA and experimentation
- `docs/` — architecture diagrams, API docs
- `tests/` — unit and integration tests

## Datasets
| Source | Cells | Key Variables |
|--------|-------|---------------|
| NASA PCoE | 18650 Li-ion | V, I, T, capacity, impedance |
| CALCE UMD | CS2 / CX2 | Multi C-rate, multi-temperature |
| Stanford MATR | 124 cells | Fast-charge protocols |
| RWTH Aachen | NMC | Real EV drive cycles |

## Tech Stack
- **Data Engineering**: PySpark, PostgreSQL, AWS S3
- **ML**: PyTorch (PINN), XGBoost, Scikit-learn, SHAP
- **App**: FastAPI + React + TailwindCSS
- **AI Modules**: OpenAI GPT-4o, HuggingFace, OpenCV
- **BI**: Power BI (DirectQuery)
