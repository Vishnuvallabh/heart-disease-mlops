#!/usr/bin/env bash
# Brings the whole project up on a fresh VM for the demo video, and shuts it down again.
#
#   bash scripts/demo.sh up       # setup + train + Docker + Minikube + monitoring (about 5-10 min the first time)
#   bash scripts/demo.sh status   # show what is running
#   bash scripts/demo.sh down     # stop everything
#
# On a VM that has just been reset:
#   git clone https://github.com/Vishnuvallabh/heart-disease-mlops.git
#   cd heart-disease-mlops && bash scripts/demo.sh up
set -euo pipefail

cd "$(dirname "$0")/.."
RUN=.demo                      # PID files and logs of the background services
K="minikube kubectl --"
NS=heart-disease
mkdir -p "$RUN"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# start_bg <name> <command>: run a command in the background unless it is already running
start_bg() {
    local name=$1; shift
    if [[ -f $RUN/$name.pid ]] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then
        echo "$name already running"
        return
    fi
    # own process group, so "down" also stops child processes (MLflow starts several)
    setsid nohup "$@" > "$RUN/$name.log" 2>&1 &
    echo $! > "$RUN/$name.pid"
    echo "$name started (log: $RUN/$name.log)"
}

stop_bg() {
    local name=$1
    if [[ -f $RUN/$name.pid ]]; then
        kill -- -"$(cat "$RUN/$name.pid")" 2>/dev/null || true
        rm -f "$RUN/$name.pid"
        echo "$name stopped"
    fi
}

wait_for() {               # wait_for <url> <seconds>
    for _ in $(seq 1 "$2"); do
        curl -s -o /dev/null "$1" && return 0
        sleep 1
    done
    echo "timed out waiting for $1" >&2
    return 1
}

up() {
    step "1/7 Python environment (venv + requirements.txt)"
    [[ -d venv ]] || python3.12 -m venv venv
    # shellcheck disable=SC1091
    source venv/bin/activate
    pip install -q -r requirements.txt
    export MLFLOW_DISABLE_AGENT_HINT=1

    step "2/7 Data + training (MLflow runs are recreated in mlflow.db)"
    python src/download_data.py
    python src/preprocess.py
    python src/train.py 2>&1 | grep -E "cv_roc_auc|Final"
    python src/predict.py sample_patient.json

    step "3/7 MLflow UI on port 5001"
    start_bg mlflow mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
    wait_for http://localhost:5001 60

    step "4/7 Docker image heart-api:1.1"
    docker build -q -t heart-api:1.1 .

    step "5/7 Minikube cluster"
    minikube status >/dev/null 2>&1 || minikube start --driver=docker
    minikube image load heart-api:1.1

    step "6/7 Deploy API + Prometheus + Grafana"
    $K apply -f k8s/namespace.yaml
    $K apply -f k8s/ -f k8s/monitoring/
    $K rollout restart deploy/heart-api -n $NS     # pick up the freshly loaded image
    $K rollout status deploy -n $NS --timeout=300s
    start_bg tunnel minikube tunnel
    # localhost ports, so the VS Code "Ports" tab can forward them to the laptop browser
    start_bg pf-api        $K port-forward -n $NS svc/heart-api 8080:8080
    start_bg pf-prometheus $K port-forward -n $NS svc/prometheus 9090:9090
    start_bg pf-grafana    $K port-forward -n $NS svc/grafana 3000:3000
    wait_for http://localhost:8080/health 60

    step "7/7 Warm-up traffic so the Grafana dashboard has data"
    python src/generate_traffic.py http://localhost:8080 100

    status
    cat <<'EOF'

Ready to record. Talk track: docs/video_script.md
Open in the browser (VS Code -> Ports tab -> forward the port if it is not listed):
  API docs     http://localhost:8080/docs
  MLflow       http://localhost:5001
  Prometheus   http://localhost:9090/targets
  Grafana      http://localhost:3000   (admin / admin -> Skip -> Dashboards -> MLOps -> Heart Disease API)
EOF
}

status() {
    step "Status"
    $K get all -n $NS 2>/dev/null || echo "cluster not running"
    for name in mlflow tunnel pf-api pf-prometheus pf-grafana; do
        if [[ -f $RUN/$name.pid ]] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then
            echo "  $name: running"
        else
            echo "  $name: stopped"
        fi
    done
}

down() {
    step "Stopping background services and Minikube"
    for name in pf-api pf-prometheus pf-grafana tunnel mlflow; do stop_bg "$name"; done
    docker rm -f heart-api >/dev/null 2>&1 || true
    minikube stop || true
}

case "${1:-}" in
    up) up ;;
    status) status ;;
    down) down ;;
    *) echo "usage: bash scripts/demo.sh {up|status|down}"; exit 1 ;;
esac
