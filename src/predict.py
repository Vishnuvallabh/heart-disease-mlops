"""Predict heart disease for one patient given as a JSON file."""
import json
import sys

import joblib
import pandas as pd

MODEL_PATH = "models/model.joblib"
METADATA_PATH = "models/metadata.json"


def load_model(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    model = joblib.load(model_path)
    with open(metadata_path) as f:
        features = json.load(f)["features"]
    return model, features


def predict(patient, model, features):
    """Return prediction (0/1), a label, and the model's confidence."""
    row = pd.DataFrame([patient])[features]
    prediction = int(model.predict(row)[0])
    confidence = float(model.predict_proba(row)[0][prediction])
    return {
        "prediction": prediction,
        "label": "disease" if prediction == 1 else "no disease",
        "confidence": round(confidence, 3),
    }


def main():
    model, features = load_model()
    with open(sys.argv[1]) as f:
        patient = json.load(f)
    print(json.dumps(predict(patient, model, features), indent=2))


if __name__ == "__main__":
    main()
