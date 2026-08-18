"""
BatteryIQ — FastAPI Backend
============================
Main application entry point.

Run from app/backend/:
  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import predict
from api import fleet
from api import chat
from api import vision
from api import nlp
from api import reports
import uvicorn

app = FastAPI(
    title="BatteryIQ API",
    description="Physics-Informed ML for EV Battery Degradation Prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS — allow React frontend ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers ────────────────────────────────────────────────────────
app.include_router(predict.router,  prefix="/predict",        tags=["Prediction"])
app.include_router(fleet.router,    prefix="/fleet",          tags=["Fleet"])
app.include_router(chat.router,     prefix="/chat",           tags=["BatteryChat"])
app.include_router(vision.router,   prefix="/analyse-image",  tags=["Vision"])
app.include_router(nlp.router,      prefix="/analyse-report", tags=["NLP"])
app.include_router(reports.router,  prefix="/generate-report",tags=["Reports"])


@app.get("/")
def root():
    return {
        "name"   : "BatteryIQ API",
        "version": "1.0.0",
        "status" : "running",
        "docs"   : "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
