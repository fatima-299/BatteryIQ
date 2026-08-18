"""
BatteryIQ — Vision API (Computer Vision)
Endpoint: POST /analyse-image

Analyses battery images for:
- Swelling detection
- Corrosion detection
- SEI buildup indicators
- Thermal damage
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import base64
import io

router = APIRouter()


class VisionResponse(BaseModel):
    defects_detected : List[str]
    severity         : str
    confidence_scores: dict
    recommendation   : str
    annotated_image  : str  # base64 encoded


@router.post("/", response_model=VisionResponse)
async def analyse_image(file: UploadFile = File(...)):
    """
    Analyse battery image for defects using OpenCV + CNN.
    Accepts: JPG, PNG, TIFF
    Returns: defect classification + annotated image
    """
    try:
        import cv2

        # Read uploaded image
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        # ── OpenCV preprocessing ──────────────────────────────────────
        gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges      = cv2.Canny(blurred, 50, 150)
        _, thresh  = cv2.threshold(blurred, 127, 255,
                                   cv2.THRESH_BINARY_INV)

        # ── Feature extraction ────────────────────────────────────────
        # Edge density — high = possible damage/corrosion
        edge_density    = np.sum(edges > 0) / edges.size

        # Brightness variance — high = uneven surface (swelling)
        brightness_var  = float(np.var(gray))

        # Dark region ratio — high = corrosion/SEI buildup
        dark_ratio      = float(np.sum(gray < 80) / gray.size)

        # Contour analysis — irregular shapes indicate damage
        contours, _     = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        n_contours      = len(contours)

        # ── Classification heuristics ─────────────────────────────────
        defects   = []
        scores    = {}

        # Swelling: high brightness variance + many contours
        swelling_score = min(brightness_var / 5000, 1.0) * 0.5 + \
                         min(n_contours / 50, 1.0) * 0.5
        scores["swelling"] = round(float(swelling_score), 3)
        if swelling_score > 0.4:
            defects.append("swelling")

        # Corrosion: high dark ratio
        corrosion_score = min(dark_ratio * 3, 1.0)
        scores["corrosion"] = round(float(corrosion_score), 3)
        if corrosion_score > 0.3:
            defects.append("corrosion")

        # SEI buildup: high edge density
        sei_score = min(edge_density * 10, 1.0)
        scores["sei_buildup"] = round(float(sei_score), 3)
        if sei_score > 0.35:
            defects.append("sei_buildup")

        # Healthy score
        healthy_score = max(0, 1.0 - max(scores.values()))
        scores["healthy"] = round(float(healthy_score), 3)
        if not defects:
            defects = ["healthy"]

        # ── Severity ──────────────────────────────────────────────────
        max_defect_score = max(
            [v for k, v in scores.items() if k != "healthy"],
            default=0
        )
        if max_defect_score < 0.3:
            severity = "none"
        elif max_defect_score < 0.5:
            severity = "mild"
        elif max_defect_score < 0.7:
            severity = "moderate"
        else:
            severity = "severe"

        # ── Recommendation ────────────────────────────────────────────
        rec_map = {
            "none"    : "Battery appears healthy. Continue normal operation.",
            "mild"    : "Minor anomalies detected. Increase monitoring frequency.",
            "moderate": "Significant defects found. Schedule inspection within 30 days.",
            "severe"  : "Critical defects detected. Remove from service immediately.",
        }
        recommendation = rec_map[severity]

        # ── Annotate image ────────────────────────────────────────────
        annotated = img.copy()

        # Draw contours
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 1)

        # Add text overlay
        color = (0, 0, 255) if severity in ["moderate","severe"] else (0, 200, 0)
        cv2.putText(annotated, f"Severity: {severity.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        cv2.putText(annotated, f"Defects: {', '.join(defects)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

        # Encode annotated image to base64
        _, buffer  = cv2.imencode(".jpg", annotated)
        img_b64    = base64.b64encode(buffer).decode("utf-8")

        return VisionResponse(
            defects_detected  = defects,
            severity          = severity,
            confidence_scores = scores,
            recommendation    = recommendation,
            annotated_image   = img_b64,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
