"""
BatteryIQ — NLP Report Analyser API
Endpoint: POST /analyse-report

Analyses maintenance logs using:
- HuggingFace BERT NER (entity extraction)
- Keyword extraction
- Sentiment/urgency classification
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi import Form
from pydantic import BaseModel
from typing import List, Optional
import re

router = APIRouter()


class NLPResponse(BaseModel):
    cell_ids_mentioned : List[str]
    keywords           : List[str]
    urgency_level      : str
    anomalies_detected : List[str]
    summary            : str
    recommendations    : List[str]


def extract_cell_ids(text: str) -> List[str]:
    """Extract battery cell IDs from text using regex patterns."""
    patterns = [
        r'\b(B\d{4})\b',               # NASA: B0005
        r'\b(CS2_\d+)\b',              # CALCE CS2
        r'\b(CX2_\d+)\b',              # CALCE CX2
        r'\b(\d{4}-\d{2}-\d{2}_c\d+)\b', # Stanford: 2017-05-12_c01
        r'\b(hnei_HNEI_[a-z])\b',     # HNEI
        # Generic: cell_XYZ / cell XYZ42 — requires an explicit separator
        # (so "cells"/"cell" alone can't match) and at least one digit in
        # the captured token (so it can't grab stray plain-English words).
        r'\bcell[_\s]+([A-Za-z0-9_]*\d[A-Za-z0-9_]*)\b',
    ]
    found = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.extend(matches)
    return list(set(found))


def extract_keywords(text: str) -> List[str]:
    """Extract battery-domain keywords."""
    battery_keywords = [
        "degradation", "capacity", "voltage", "temperature",
        "resistance", "SOH", "RUL", "cycle", "charging",
        "discharging", "swelling", "corrosion", "overheating",
        "failure", "replacement", "maintenance", "inspection",
        "thermal", "electrolyte", "anode", "cathode", "SEI",
        "lithium", "NMC", "LFP", "battery", "cell", "fault",
        "anomaly", "warning", "critical", "urgent", "normal",
    ]
    text_lower = text.lower()
    found      = [kw for kw in battery_keywords if kw.lower() in text_lower]
    return list(set(found))


def classify_urgency(text: str, keywords: List[str]) -> str:
    """Classify urgency level of maintenance report."""
    critical_words = ["critical", "urgent", "immediate", "failure",
                      "broken", "fire", "smoke", "danger", "replace now"]
    warning_words  = ["warning", "concern", "unusual", "abnormal",
                      "inspect", "monitor", "degrading"]
    normal_words   = ["normal", "routine", "scheduled", "ok",
                      "good", "healthy"]

    text_lower = text.lower()
    if any(w in text_lower for w in critical_words):
        return "critical"
    elif any(w in text_lower for w in warning_words):
        return "warning"
    elif any(w in text_lower for w in normal_words):
        return "normal"
    else:
        return "unknown"


def detect_anomalies(text: str) -> List[str]:
    """Detect specific battery anomalies mentioned in text."""
    anomaly_map = {
        "swelling"         : ["swell", "bloat", "bulg", "expand"],
        "overheating"      : ["hot", "heat", "overheat", "temperature high", "thermal"],
        "capacity fade"    : ["capacity drop", "range loss", "shorter range", "capacity fade"],
        "voltage drop"     : ["voltage drop", "low voltage", "voltage issue"],
        "internal short"   : ["short circuit", "spark", "internal short"],
        "electrolyte leak" : ["leak", "electrolyte", "liquid"],
        "corrosion"        : ["rust", "corrosi", "oxidat"],
    }
    text_lower  = text.lower()
    detected    = []
    for anomaly, triggers in anomaly_map.items():
        if any(t in text_lower for t in triggers):
            detected.append(anomaly)
    return detected


def generate_recommendations(urgency: str, anomalies: List[str]) -> List[str]:
    """Generate maintenance recommendations."""
    recs = []
    if urgency == "critical":
        recs.append("Remove affected cells from service immediately")
        recs.append("Conduct full safety inspection before redeployment")
    elif urgency == "warning":
        recs.append("Schedule inspection within 7 days")
        recs.append("Increase monitoring frequency to daily")

    if "overheating" in anomalies:
        recs.append("Check thermal management system")
        recs.append("Reduce charging rate to 0.5C")
    if "swelling" in anomalies:
        recs.append("Do not charge swollen cells — fire risk")
    if "capacity fade" in anomalies:
        recs.append("Compare against SOH baseline in BatteryIQ dashboard")
    if "corrosion" in anomalies:
        recs.append("Clean terminals and inspect for moisture ingress")

    if not recs:
        recs.append("Continue regular monitoring schedule")

    return recs


@router.post("/", response_model=NLPResponse)
async def analyse_report(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Analyse a maintenance report text or file.
    Extracts cell IDs, keywords, anomalies, and generates recommendations.
    """
    try:
        # Get text from form or file
        if file:
            content = await file.read()
            text    = content.decode("utf-8", errors="ignore")
        elif not text:
            raise HTTPException(
                status_code=400,
                detail="Provide either text or file"
            )

        # Run NLP pipeline
        cell_ids  = extract_cell_ids(text)
        keywords  = extract_keywords(text)
        urgency   = classify_urgency(text, keywords)
        anomalies = detect_anomalies(text)
        recs      = generate_recommendations(urgency, anomalies)

        # Generate summary
        n_cells   = len(cell_ids)
        n_anomaly = len(anomalies)
        summary   = (
            f"Report analysis complete. "
            f"Found {n_cells} cell reference(s) "
            f"({'named: ' + ', '.join(cell_ids[:3]) if cell_ids else 'no specific cells'}"
            f"{', ...' if n_cells > 3 else ''}). "
            f"Urgency level: {urgency.upper()}. "
            f"{'Anomalies detected: ' + ', '.join(anomalies) + '.' if anomalies else 'No specific anomalies detected.'}"
        )

        # Try HuggingFace NER for enhanced extraction
        try:
            from transformers import pipeline
            ner = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
            entities = ner(text[:512])  # limit to 512 tokens
            org_entities = [
                e["word"] for e in entities
                if e["entity_group"] in ["ORG", "LOC", "MISC"]
                and len(e["word"]) > 2
            ]
            keywords = list(set(keywords + org_entities[:5]))
        except Exception:
            pass  # HuggingFace optional — fallback to regex

        return NLPResponse(
            cell_ids_mentioned = cell_ids,
            keywords           = keywords[:15],
            urgency_level      = urgency,
            anomalies_detected = anomalies,
            summary            = summary,
            recommendations    = recs,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
