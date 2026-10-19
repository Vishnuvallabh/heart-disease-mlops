"""FastAPI service that serves the trained heart disease model."""
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import load_model, predict

model, features = load_model()

app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts heart disease risk from 13 patient features "
                "(UCI Cleveland dataset).",
    version="1.0.0",
)


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


@app.post("/predict", response_model=Prediction)
def predict_endpoint(patient: Patient):
    """Return 0/1 prediction, a label and the model's confidence."""
    return predict(patient.model_dump(), model, features)
