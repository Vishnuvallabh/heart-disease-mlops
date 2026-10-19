# Demo video: talk track (about 8 minutes)

**Before recording:** run `bash scripts/demo.sh up` and wait for "Ready to record".

**Setup:**
- Open these browser tabs: GitHub repo, GitHub Actions, API `/docs`, MLflow, Prometheus, Grafana.
- In VS Code, open one terminal inside the repo and run `source venv/bin/activate`.

Each section lists what to **show** and what to **say**.

---

## 0. Intro (30 s)
**Show:** the GitHub repo front page with the README and architecture diagram.

**Say:** "This is my MLOps assignment: a heart-disease risk classifier on the UCI Cleveland dataset. I'll walk through the whole pipeline: data, training with MLflow tracking, tests and CI, Docker, Kubernetes deployment and monitoring. I'm recording on the Prayogshala VM, from a fresh clone of this repo."

## 1. Data and EDA (1 min)
**Show:** the terminal and `notebooks/01_eda.ipynb` (class balance, histograms, correlation heatmap).

```bash
python src/download_data.py
python src/preprocess.py
```

**Say:**
- "The download script fetches the raw Cleveland file from UCI."
- "Preprocessing fills the few missing `ca` and `thal` values with the mode, and turns the 0–4 `num` column into a binary target: disease or no disease."
- "That gives 303 rows. The classes are roughly balanced, so accuracy is meaningful, but I also track recall and ROC-AUC."

## 2. Features and models (1 min)
**Show:** `src/features.py`, then `src/train.py`.

```bash
python src/train.py
```

**Say:**
- "Features are in three groups: numeric columns are standardised, categorical columns like chest-pain type are one-hot encoded, and binary columns pass through."
- "The preprocessing is part of the scikit-learn pipeline, so training and serving use exactly the same transformations."
- "I train Logistic Regression and Random Forest, each tuned with GridSearchCV over 5-fold stratified cross-validation."
- "The final model is picked by cross-validated ROC-AUC, not by the test score, to avoid overfitting to the test set."
- "Logistic Regression wins with 0.907 CV ROC-AUC and 0.958 on the test set."

## 3. Experiment tracking: MLflow (1 min)
**Show:** MLflow at `localhost:5001`, in "Model training" mode → `heart-disease` → Runs. Then:
1. Expand a parent run to show its grid-search child runs.
2. Open the `logistic_regression` run: parameters, then Model metrics, then Artifacts (ROC curve, confusion matrix).

**Say:** "Every model is a parent run, and every grid-search candidate is a child run. Parameters, metrics, plots and the model itself are all logged."

## 4. Packaging and reproducibility (30 s)
**Show:** the `models/` folder and `requirements.txt`.

```bash
python src/predict.py sample_patient.json
```

**Say:** "The whole pipeline, preprocessing plus model, is saved as one joblib file with a metadata file. Versions are pinned in `requirements.txt`, so a clean install gives the same results."

## 5. Tests and CI/CD (1.5 min)
**Show:** the terminal.

```bash
flake8 src tests && pytest -v
```

Then open GitHub → Actions → the latest green run:
1. The four jobs.
2. The pytest log and the training log.
3. The `test-results` and `model-and-reports` artifacts at the bottom.

**Say:**
- "There are 18 unit tests covering data cleaning, the model and the API."
- "The GitHub Actions pipeline runs lint, then tests, then trains the model and builds and tests the Docker image."
- "Each job depends on the one before it, so a lint or test failure stops the pipeline." Optionally show the earlier failed run.

## 6. Docker (1 min)
**Show:** the `Dockerfile`, then the terminal.

```bash
docker build -t heart-api:1.1 .
docker run -d -p 8000:8000 --name heart-api heart-api:1.1
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_patient.json
docker rm -f heart-api
```

**Say:** "The image is python:3.12-slim with only the API dependencies. It runs as a non-root user and has a health check. `/predict` takes JSON and returns the prediction and its confidence."

## 7. Kubernetes deployment (1.5 min)
**Show:** `k8s/deployment.yaml` and `k8s/service.yaml`, then the terminal.

```bash
minikube kubectl -- get all -n heart-disease
curl -X POST localhost:8080/predict -H "Content-Type: application/json" -d @sample_patient.json
```

Then open `/docs` → `/predict` → Try it out.

**Say:** "Two replicas, with readiness and liveness probes, behind a LoadBalancer service. On this VM it's Minikube, and `minikube tunnel` gives the LoadBalancer an IP."

**Optional self-healing demo:**

```bash
minikube kubectl -- delete pod -n heart-disease -l app=heart-api --wait=false
minikube kubectl -- get pods -n heart-disease -w
```

## 8. Monitoring and logging (1 min)
**Show:** the terminal.

```bash
python src/generate_traffic.py http://localhost:8080 200
minikube kubectl -- logs -n heart-disease -l app=heart-api --tail=5
```

Then:
1. Prometheus → Status → Target health: both pods UP.
2. Grafana → Heart Disease API dashboard.

**Say:**
- "Every request is logged as JSON with its path, status and latency."
- "The API exposes Prometheus metrics: request count, latency histogram, predictions by result and confidence."
- "The dashboard shows requests per second, latency p50/p95, error rate and requests per pod. The 422s are deliberately invalid inputs from the traffic script."

## 9. Wrap-up (20 s)
**Say:** "So from one `git push`, the code is linted, tested, trained and containerised in CI, then deployed to Kubernetes with monitoring. All screenshots and the report are in the repo."

**After recording:** run `bash scripts/demo.sh down`.
