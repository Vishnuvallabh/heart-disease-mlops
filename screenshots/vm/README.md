# Evidence: full pipeline re-run on a Linux VM

The project was re-run from a fresh `git clone` on a second machine, separate from the Mac laptop where it was built, to show that it is reproducible.

- **Host:** AWS EC2 `ip-10-3-16-189.ap-south-1.compute.internal`
- **OS:** RHEL 9 (kernel 5.14.0-503)
- **Python:** 3.12.9
- **Docker:** 28.0.1
- **Kubernetes:** Minikube, Kubernetes v1.32.0

Every terminal capture starts with a `### user@host | timestamp` line. `logs/` holds the raw text, and `terminal/` holds the same output rendered as images.

**Differences from the laptop setup:**
- Kubernetes runs on **Minikube** instead of Docker Desktop.
- The image is loaded into the cluster with `minikube image load heart-api:1.1`. The manifest sets `imagePullPolicy: Never`.
- `minikube tunnel` gives the LoadBalancer services their external IPs. The `k8s/` manifests are used unchanged.

| Task | Evidence |
|---|---|
| Clean setup from requirements | `terminal/09_clean_setup.png`: fresh clone, venv, `pip install -r requirements.txt`, pipeline and 18 tests pass |
| 1. Data & EDA | `terminal/01_data_preprocess.png`, `terminal/01_eda_notebook.png`, `eda/*.png`, `01_eda_executed.ipynb` |
| 2. Models & evaluation | `terminal/02_train.png`: LR CV ROC-AUC 0.907 / test ROC-AUC 0.958; RF 0.894 / 0.947 |
| 3. MLflow tracking | `mlflow_runs.png`, `mlflow_run_details.png`, `mlflow_model_metrics.png`, `mlflow_artifacts.png`, `terminal/03_mlflow_runs.png` |
| 4. Packaging | `terminal/04_packaging.png`: `model.joblib` + `metadata.json`, CLI prediction |
| 5. Lint & tests | `terminal/05_lint_tests.png`: flake8 clean, 18/18 pytest pass |
| 6. Docker | `terminal/06_docker_build.png`, `terminal/06_docker_run.png`: `/health` + `/predict` in the container |
| 7. Kubernetes | `terminal/07_k8s_get_all.png`, `terminal/07_k8s_predict.png`, `terminal/07_k8s_scaling.png`, `terminal/07_k8s_self_healing.png`, `api_swagger_docs.png`, `api_swagger_predict.png` (live call through the LoadBalancer) |
| 8. Monitoring & logging | `prometheus_targets.png`, `prometheus_graph.png`, `grafana_dashboard.png`, `terminal/08_traffic.png`, `terminal/08_api_request_logs.png`, `terminal/08_prometheus_query.png` |
