from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
          "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
          "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_sample_patient():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 0
    assert body["label"] == "no disease"
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_rejects_missing_field():
    patient = {k: v for k, v in SAMPLE.items() if k != "age"}
    assert client.post("/predict", json=patient).status_code == 422


def test_predict_rejects_out_of_range_value():
    assert client.post("/predict", json={**SAMPLE, "age": 500}).status_code == 422
