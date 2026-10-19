"""Send random patient requests to the API so the dashboard has data.

Usage: python src/generate_traffic.py [base_url] [n_requests]
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300


def random_patient():
    return {
        "age": random.randint(29, 77), "sex": random.randint(0, 1),
        "cp": random.randint(1, 4), "trestbps": random.randint(94, 200),
        "chol": random.randint(126, 564), "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2), "thalach": random.randint(71, 202),
        "exang": random.randint(0, 1), "oldpeak": round(random.uniform(0, 6.2), 1),
        "slope": random.randint(1, 3), "ca": random.randint(0, 3),
        "thal": random.choice([3, 6, 7]),
    }


def send(patient):
    req = urllib.request.Request(
        f"{BASE_URL}/predict", data=json.dumps(patient).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


def main():
    counts = {}
    for i in range(N):
        patient = random_patient()
        if random.random() < 0.1:          # ~10% deliberately invalid -> 422
            patient["age"] = 500
        status = send(patient)
        counts[status] = counts.get(status, 0) + 1
        time.sleep(random.uniform(0.05, 0.3))
    print(f"Sent {N} requests to {BASE_URL}: {counts}")


if __name__ == "__main__":
    main()
