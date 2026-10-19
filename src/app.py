"""FastAPI service that serves the trained heart disease model."""
import json
import logging
import time
from typing import Literal

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from predict import load_model, predict

# ---------- logging: one JSON line per request, printed to stdout ----------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("heart-api")

# ---------- Prometheus metrics ----------
REQUESTS = Counter(
    "api_requests_total", "HTTP requests received",
    ["method", "endpoint", "status"])
LATENCY = Histogram(
    "api_request_latency_seconds", "Time taken to answer a request",
    ["endpoint"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
PREDICTIONS = Counter(
    "api_predictions_total", "Predictions returned, by result", ["label"])
CONFIDENCE = Histogram(
    "api_prediction_confidence", "Confidence of returned predictions",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))

model, features = load_model()

app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts heart disease risk from 13 patient features "
                "(UCI Cleveland dataset).",
    version="1.1.0",
)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    """Time every request, count it, and log it (except /metrics and /health)."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    path = request.url.path

    if path != "/metrics":
        REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        LATENCY.labels(path).observe(duration)
    if path not in ("/metrics", "/health"):
        logger.info(json.dumps({
            "event": "request",
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "latency_ms": round(duration * 1000, 2),
            "client": request.client.host if request.client else None,
        }))
    return response


class Patient(BaseModel):
    """One patient's features. Ranges reject obviously invalid input."""
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)")
    ca: int = Field(..., ge=0, le=3, description="Major vessels coloured by fluoroscopy (0-3)")
    thal: Literal[3, 6, 7] = Field(..., description="3 = normal, 6 = fixed, 7 = reversible defect")

    model_config = {
        "json_schema_extra": {
            "example": {"age": 63, "sex": 1, "cp": 1, "trestbps": 145,
                        "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
                        "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0,
                        "thal": 6}
        }
    }


class Prediction(BaseModel):
    prediction: int
    label: str
    confidence: float


@app.get("/health")
def health():
    """Liveness check used by Docker and Kubernetes."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Prometheus scrapes this page every few seconds."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=Prediction)
def predict_endpoint(patient: Patient):
    """Return 0/1 prediction, a label and the model's confidence."""
    result = predict(patient.model_dump(), model, features)
    PREDICTIONS.labels(result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])
    logger.info(json.dumps({"event": "prediction", **result}))
    return result
