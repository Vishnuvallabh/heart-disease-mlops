# Demo video script – Mac (Docker Desktop Kubernetes)

About **12 minutes**, one section per assignment task. Each section shows:
- **OPEN**: what to put on screen
- **DO**: what to click or type
- **SAY**: what to say, word for word or in your own words
- **POINT AT**: the result to show with the mouse

If you need to be shorter, skip the parts marked *(optional)*.

---

## A. Before you press record (10 min, not recorded)

### 1. Wake up Docker Desktop
Open the **Docker Desktop** app. If it says *Resource Saver* or *Paused*, click **Resume** or **Start**. Wait until the bottom left shows **Engine running** and **Kubernetes running** (both green).

### 2. Terminal setup
In VS Code, open the terminal and run this. Keep this terminal open for the whole video.

```bash
cd ~/Desktop/MLOPS
source venv/bin/activate
export PATH="$HOME/.docker/bin:$PATH"
export MLFLOW_DISABLE_AGENT_HINT=1
kubectl get pods -n heart-disease          # expect 4 pods Running (2 heart-api, prometheus, grafana)
curl http://localhost:8080/health          # expect {"status":"ok"}
```

If the pods are missing, deploy again:
```bash
kubectl apply -f k8s/namespace.yaml && kubectl apply -f k8s/ && kubectl apply -f k8s/monitoring/
```

### 3. Start MLflow in a second terminal
Click **+** in the terminal panel, then run:
```bash
cd ~/Desktop/MLOPS && source venv/bin/activate && mlflow ui --port 5001
```
Leave it running and switch back to the first terminal.

### 4. Warm up Grafana
```bash
python src/generate_traffic.py http://localhost:8080 100
```

### 5. Open these Chrome tabs, in this order
1. https://github.com/Vishnuvallabh/heart-disease-mlops (repo front page)
2. https://github.com/Vishnuvallabh/heart-disease-mlops/actions
3. http://localhost:5001 (MLflow) → **Model training** → **heart-disease**
4. http://localhost:8080/docs (API)
5. http://localhost:9090/targets (Prometheus)
6. http://localhost:3000 (Grafana, log in as `admin` / `admin`, then **Skip**) → Dashboards → MLOps → **Heart Disease API**

### 6. Make the recording readable
- VS Code: press **Cmd +** two or three times so the terminal text is large.
- Chrome: zoom to 110–125% (**Cmd +**).
- Close Slack, mail and other apps, and turn on **Do Not Disturb** in Control Center.
- Run `clear` in the terminal.

### 7. Start recording
Press **Cmd + Shift + 5** → **Record Entire Screen** → **Options** → **Microphone: MacBook Microphone** → **Record**.

---

## 0. Introduction (45 s)

**OPEN:** Chrome tab 1, the GitHub repo front page.

**SAY:**
> "Hello, I am Sri Vishnu Vallabh Yathavakilla, roll number 2025AE05312. This is my MLOps Assignment 1 for course AIMLCZG523.
> The problem is predicting heart disease risk from 13 medical measurements, using the UCI Heart Disease Cleveland dataset with 303 patients.
> I built a complete MLOps pipeline around the model: data download and cleaning, EDA, training with experiment tracking in MLflow, packaging, unit tests and a CI/CD pipeline in GitHub Actions, a Docker container, deployment on Kubernetes, and monitoring with Prometheus and Grafana.
> Everything is in this public GitHub repository."

**DO:** scroll down slowly through the README. Pause on the **CI badge** (green "passing") and the **Results** table.

---

## 1. Architecture (45 s)

**OPEN:** keep scrolling the README to the **architecture diagram**.

**POINT AT:** the three boxes as you name them.

**SAY:**
> "The architecture has three stages.
> Stage one is my laptop: download, preprocess, train two models, and log every run in MLflow. The best model is saved as one file.
> Stage two is GitHub Actions: on every push it runs lint, unit tests, retrains the model, and builds and tests the Docker image.
> Stage three is Kubernetes on Docker Desktop: the API runs as two pods behind a LoadBalancer on port 8080, Prometheus collects metrics from both pods every 5 seconds, and Grafana shows a dashboard."

---

## 2. Task 1: Data acquisition and EDA (1 min 15 s)

**OPEN:** VS Code → `src/download_data.py`, then the terminal.

**DO:**
```bash
clear
python src/download_data.py
python src/preprocess.py
```

**POINT AT:** `Downloaded to data/raw/processed.cleveland.data` and `Saved 303 rows to data/processed/heart_clean.csv`.

**SAY:**
> "The download script fetches the raw file directly from the UCI repository, so anyone can reproduce the data.
> The preprocessing script adds the column names, reads question marks as missing values, fills the 6 missing values in `ca` and `thal` with the most frequent value because they are categories, and turns the target `num` from 0–4 into binary: disease or no disease."

**OPEN:** `notebooks/01_eda.ipynb`. Scroll to each plot.

**SAY at the class-balance plot:**
> "164 patients without disease and 139 with disease, 54 to 46 percent. The classes are roughly balanced, so no resampling is needed. Always guessing 'no disease' would give 54 percent, which is the baseline to beat."

**SAY at the histograms:**
> "Most patients are 48 to 61 years old. Cholesterol has outliers up to 564, oldpeak has a spike at zero, and the scales are very different, which is why I scale the numeric features."

**SAY at the correlation heatmap** (point at the `target` row):
> "In the target row, thal, ca, exang, oldpeak and chest pain type have the strongest positive correlation with disease. Maximum heart rate, thalach, is negative at minus 0.42. Cholesterol and fasting blood sugar are close to zero."

**POINT AT:** the Markdown observation cells under each plot.

---

## 3. Task 2: Feature engineering and model development (1 min 15 s)

**OPEN:** `src/features.py`.

**SAY:**
> "Features are in three groups. Six numeric columns are standardised with StandardScaler. Four categorical columns, like chest pain type, are one-hot encoded. Three binary columns are kept as they are.
> This preprocessing is inside a scikit-learn Pipeline together with the model, so the scaler learns only from training data, and the API applies exactly the same steps."

**OPEN:** `src/train.py` → scroll to `get_models()`.

**SAY:**
> "I train two models: Logistic Regression and Random Forest. The data is split 80/20 with stratification. Each model is tuned with GridSearchCV over 5-fold stratified cross-validation: C for Logistic Regression, and number of trees and depth for Random Forest."

**DO:**
```bash
clear
python src/train.py 2>&1 | grep -E "cv_roc_auc|Final"
```

**POINT AT:** the two result lines and `Final model: logistic_regression`.

**SAY:**
> "Logistic Regression: cross-validated ROC-AUC 0.907, test accuracy 0.869, recall 0.929, test ROC-AUC 0.958. Random Forest: 0.895 CV ROC-AUC, accuracy 0.885, recall 0.964.
> I pick the final model by cross-validated ROC-AUC, not by the test score, so the test set stays a fair, unseen check. Logistic Regression wins, and it is also easier to explain.
> Recall matters most here, because missing a sick patient is worse than a false alarm."

---

## 4. Task 3: Experiment tracking with MLflow (1 min 15 s)

**OPEN:** Chrome tab 3, MLflow → **heart-disease** → **Runs**. The newest runs, from the training you just ran, are at the top.

**DO:** click **+** next to the top `logistic_regression` and `random_forest` runs to expand them.

**SAY:**
> "Every training creates 12 MLflow runs: one parent run per model, and one child run for every hyperparameter combination tried, 4 for Logistic Regression and 6 for Random Forest."

**DO:** click the `random_forest` parent run.

**POINT AT:** the **Parameters** (`max_depth`, `n_estimators`, `cv_folds`) and the **Metrics** (`test_recall`, `test_roc_auc` and so on), then **Logged models (1)**.

**SAY:**
> "Each run logs its parameters, its metrics, and the trained model itself."

**DO:** click the **Artifacts** tab → `random_forest_confusion_matrix.png`, then `random_forest_roc_curve.png`.

**SAY:**
> "The plots are logged too. The confusion matrix shows 27 healthy and 27 sick patients correct, 6 false alarms and only 1 missed sick patient."

**DO:** go back to Runs, tick the two parent runs, and click **Compare**. Scroll to Parameters and Metrics.

**SAY:**
> "The compare view shows both models side by side, which is how I made the final choice."

---

## 5. Task 4: Model packaging and reproducibility (45 s)

**OPEN:** VS Code → `models/metadata.json`, then `requirements.txt`.

**SAY:**
> "The final model is saved as one joblib file containing the whole pipeline, preprocessing plus model, so new data gets exactly the same treatment. The metadata file stores the feature order, the metrics and the scikit-learn version.
> All library versions are pinned in requirements.txt. I tested a completely fresh environment built only from this file, and a fresh clone from GitHub, and both gave identical results."

**DO:**
```bash
clear
python src/predict.py sample_patient.json
```

**POINT AT:** `"prediction": 0, "label": "no disease", "confidence": 0.823`.

*(optional)* **OPEN:** `screenshots/clean_setup_test.png` for the clean-environment proof.

---

## 6. Task 5: CI/CD pipeline and automated testing (1 min 45 s)

**OPEN:** the terminal.

**DO:**
```bash
clear
flake8 src tests && echo "flake8: no problems"
pytest -v
```

**POINT AT:** the list of test names, then `18 passed`.

**SAY:**
> "There are 18 unit tests: 5 for data cleaning, 7 for the model code, and 6 for the API. They use small fake data, so they are fast and need no internet. flake8 checks the code style."

**OPEN:** Chrome tab 2, GitHub **Actions**.

**SAY:**
> "Every push runs this GitHub Actions pipeline. All runs are green."

**DO:** click the **top run**.

**POINT AT:** the 4 connected jobs: **Lint → Unit tests → Train model** and **Build & test Docker image**.

**SAY:**
> "Lint runs first, then the unit tests. Only if they pass does it train the model and build and test the Docker image. Cheap checks first, so we never train broken code."

**DO:** click **Unit tests (pytest)** → expand **Run tests** → scroll to `18 passed`.
Then click **Train model** → expand **Train and select model** → point at the metrics and `Final model`.

**DO:** click **Summary** and scroll to the bottom.

**POINT AT:** **Artifacts**: `model-and-reports` and `test-results`.

**SAY:**
> "Every run saves artifacts: the test report, and the trained model, plots and training log."

**DO:** in the Actions list, scroll down and open run **#2 "Demo: deliberately failing test"**.

**POINT AT:** Lint ✅, Unit tests ❌, Train model *skipped*. Click Unit tests → the `FAILED ... test_clean_makes_binary_target` line.

**SAY:**
> "To prove the pipeline fails on errors, I broke a test on purpose. Lint passed, the tests failed with a clear log showing exactly which test and which values, and training was skipped. Then I reverted it and the pipeline went green again."

---

## 7. Task 6: Containerization (1 min)

**OPEN:** VS Code → `Dockerfile`.

**SAY:**
> "The Dockerfile uses the small python:3.12-slim image, installs only the API libraries from requirements-api.txt, copies only the code and the model, runs as a non-root user for security, and has a health check."

**OPEN:** `src/app.py`, briefly.

**SAY:**
> "The API is FastAPI. /predict accepts patient JSON and returns the prediction and the confidence. Every field is validated, so bad input gets a clear 422 error instead of a crash."

**DO:**
```bash
clear
docker build -t heart-api:1.1 .
docker run -d -p 8000:8000 --name heart-api heart-api:1.1
sleep 5
curl http://localhost:8000/health; echo
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_patient.json; echo
curl -s -o /dev/null -w "invalid input -> HTTP %{http_code}\n" -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age": 500}'
docker rm -f heart-api
```

**POINT AT:** `{"status":"ok"}`, `0 / no disease / 0.823`, and `HTTP 422`.

**SAY:**
> "The container builds, answers the health check, predicts the sample patient, and rejects invalid input with 422. The same build-and-test runs in GitHub Actions on every push."

---

## 8. Task 7: Production deployment on Kubernetes (1 min 30 s)

**OPEN:** VS Code → `k8s/deployment.yaml`.

**POINT AT:** `replicas: 2`, `RollingUpdate` with `maxUnavailable: 0`, `resources`, `readinessProbe`, `livenessProbe`.

**SAY:**
> "This is the Kubernetes deployment manifest. Two replicas, so one can fail while the other serves. CPU and memory limits. Readiness and liveness probes on /health. And rolling updates that replace one pod at a time, so there is no downtime."

**OPEN:** `k8s/service.yaml`.

**SAY:**
> "The Service is type LoadBalancer. Docker Desktop exposes it on localhost port 8080."

**DO:**
```bash
clear
kubectl get all -n heart-disease
```

**POINT AT:** the 2 `heart-api` pods `Running 1/1`, then `service/heart-api LoadBalancer localhost 8080`.

**DO:**
```bash
curl http://localhost:8080/health; echo
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d @sample_patient.json; echo
```

**OPEN:** Chrome tab 4, `localhost:8080/docs` → **POST /predict** → **Try it out** → **Execute**.

**POINT AT:** `Code 200` and the response body with `0.823`.

**DO (self-healing):**
```bash
clear
kubectl get pods -n heart-disease -l app=heart-api
kubectl delete $(kubectl get pods -n heart-disease -l app=heart-api -o name | head -1) -n heart-disease
kubectl get pods -n heart-disease -l app=heart-api -w
```
Wait until a new pod shows `Running 1/1`, then press **Ctrl + C**.

**SAY:**
> "To show self-healing, I delete one pod. Kubernetes immediately creates a new pod with a new name, and the other pod keeps serving traffic the whole time."

*(optional)* **DO (scaling):**
```bash
kubectl scale deployment heart-api --replicas=3 -n heart-disease && sleep 10 && kubectl get pods -n heart-disease -l app=heart-api
kubectl scale deployment heart-api --replicas=2 -n heart-disease
```
**SAY:** "And it scales to three copies with one command."

---

## 9. Task 8: Monitoring and logging (1 min 30 s)

**DO:**
```bash
clear
python src/generate_traffic.py http://localhost:8080 200
```

**SAY, while it runs:**
> "This script sends 200 random patients to the API. About 10 percent are made invalid on purpose, so we can see errors on the dashboard."

**OPEN:** Chrome tab 6, the **Grafana** dashboard. Wait for the charts to rise.

**POINT AT** each panel:
> "API pods up: 2. Total predictions keeps going up. The error rate is about 10 percent, which is exactly the deliberately invalid requests. Average confidence is about 83 percent.
> Requests per second by status: green for 200 OK, yellow for 422.
> Latency p50 and p95: the 95th percentile is around 20 to 25 milliseconds.
> Predictions per second by result, disease versus no disease.
> And requests per pod: both pods share the load."

**OPEN:** Chrome tab 5, **Prometheus** → Status → **Target health**.

**POINT AT:** `heart-api 2 / 2 up`.

**SAY:**
> "Prometheus finds the API pods automatically through Kubernetes and scrapes their /metrics page every 5 seconds. Because it discovers pods by label, the new pod from the self-healing demo was picked up automatically."

*(optional)* **DO:** **Query** tab → type `api_requests_total` → **Execute**.

**OPEN:** the terminal (after the traffic script has finished).

**DO:**
```bash
clear
kubectl logs -n heart-disease -l app=heart-api --prefix --tail=200 | grep '"event"' | tail -8
```

**POINT AT:** a `"event": "request"` line (`status`, `latency_ms`) and a `"event": "prediction"` line (`label`, `confidence`).

**SAY:**
> "Every request is logged as one JSON line with path, status and latency, and every prediction with its label and confidence. JSON logs are easy to search and filter."

---

## 10. Task 9: Documentation and wrap-up (30 s)

**OPEN:** the GitHub repo → `report/` folder, then back to the README.

**SAY:**
> "The repository has the README with setup and access instructions, the architecture diagram, all screenshots, and the full report in the report folder.
> To summarise: one git push lints, tests, trains and containerises the model automatically, and the model runs on Kubernetes as a self-healing, monitored service with request logging and a Grafana dashboard.
> Thank you."

**DO:** stop the recording. Click the ■ icon in the menu bar, or press **Cmd + Ctrl + Esc**.

---

## B. After recording

1. Watch it once. Check that the audio is clear and the terminal text is readable.
2. Upload to Google Drive or YouTube as **Unlisted**. Set sharing to **Anyone with the link**.
3. Send the link to Claude to add it to the README and the report.
4. Optional cleanup when grading is over: `kubectl delete namespace heart-disease`.

## C. If something goes wrong while recording

| Problem | Fix |
|---|---|
| `kubectl` "TLS handshake timeout" or "connection refused" | Docker Desktop is paused. Click **Resume** in Docker Desktop and wait for Kubernetes to turn green. |
| `kubectl: command not found` | `export PATH="$HOME/.docker/bin:$PATH"` |
| `curl localhost:8080` refused | `kubectl get pods -n heart-disease`. If none, rerun the apply commands in A.2. |
| Grafana panels say "No data" | Run the traffic script again and wait about 30 seconds. Check the time range is "Last 15 minutes". |
| `docker run` says the name is in use | `docker rm -f heart-api`, then run it again. |
| MLflow page does not load | Check that the second terminal still shows `mlflow ui` running. If not, start it again. |

Small mistakes are fine. Pause, fix it, and continue. You can also cut mistakes out with QuickTime's **Edit → Trim**.
