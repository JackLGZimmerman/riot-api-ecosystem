#!/usr/bin/env bash

set -euo pipefail

compose_down_args=(--remove-orphans)
cleanup_message="Cancelled during restart cleanup before launching a new deployment run."
completion_message="Restart complete"

if [ "${1:-}" = "--fresh" ]; then
  compose_down_args=(-v --remove-orphans)
  cleanup_message="Cancelled during fresh-start cleanup before launching a new deployment run."
  completion_message="Fresh start complete"
elif [ "$#" -ne 0 ]; then
  echo "Usage: $0 [--fresh]" >&2
  exit 2
fi

clear_stale_prefect_runs() {
  docker exec \
    -e CANCEL_MESSAGE="$cleanup_message" \
    prefect-server \
    python - <<'PY'
import os
import requests

base = "http://localhost:4200/api"
deployment_name = "riot-pipeline/riot-pipeline"
cancel_message = os.environ["CANCEL_MESSAGE"]

deployment = requests.get(f"{base}/deployments/name/{deployment_name}", timeout=30)
if deployment.status_code == 404:
    print("No existing deployment found; skipping stale-run cleanup.")
    raise SystemExit(0)
deployment.raise_for_status()
deployment_id = deployment.json()["id"]

runs = requests.post(
    f"{base}/flow_runs/filter",
    json={"flow_runs": {"state": {"type": {"any_": ["RUNNING", "PENDING", "SCHEDULED", "CANCELLING"]}}}},
    timeout=60,
)
runs.raise_for_status()

stale_runs = [run for run in runs.json() if run.get("deployment_id") == deployment_id]
if not stale_runs:
    print("No non-terminal runs found for deployment.")
    raise SystemExit(0)

for run in stale_runs:
    requests.post(
        f"{base}/flow_runs/{run['id']}/set_state",
        json={
            "state": {
                "type": "CANCELLED",
                "name": "Cancelled",
                "message": cancel_message,
            },
            "force": True,
        },
        timeout=60,
    ).raise_for_status()
    print(f"Cancelled stale flow run {run['id']} ({run['state_type']}/{run['state_name']}).")
PY
}

echo "========================================"
echo "Ensuring local log files are writable..."
echo "========================================"
mkdir -p app/core/logging/logs/schema_drift
for path in \
  app/core/logging/logs/app.log.jsonl \
  app/core/logging/logs/schema_drift/non_timeline.log.jsonl \
  app/core/logging/logs/schema_drift/timeline.log.jsonl
do
  [ ! -e "$path" ] || [ -w "$path" ] || rm -f "$path"
  touch "$path"
  chmod u+rw "$path"
done

echo "========================================"
echo "Attempting graceful pipeline stop..."
echo "========================================"
./stop_pipeline_safely.sh || true

echo "========================================"
if [ "${1:-}" = "--fresh" ]; then
  echo "Fresh start: stopping containers and removing volumes..."
else
  echo "Restart: stopping containers (preserving volumes)..."
fi
echo "========================================"
docker compose down "${compose_down_args[@]}"

echo "========================================"
echo "Cleaning old Prefect flow-run containers..."
echo "========================================"
docker ps -aq --filter label=io.prefect.flow-run-id | xargs -r docker rm -f

echo "========================================"
echo "Starting containers with rebuild and waiting for health..."
echo "========================================"
docker compose up -d --build --wait

echo "========================================"
echo "Clearing stale Prefect runs for deployment..."
echo "========================================"
clear_stale_prefect_runs

echo "========================================"
echo "Building riot-pipeline image (no cache)..."
echo "========================================"
docker build --no-cache -t riot-pipeline:latest .

echo "========================================"
echo "Deploying Prefect flow..."
echo "========================================"
.venv/bin/prefect --no-prompt deploy --prefect-file prefect.yaml

echo "========================================"
echo "Running Prefect deployment..."
echo "========================================"
.venv/bin/prefect deployment run 'riot-pipeline/riot-pipeline'

echo "========================================"
echo "$completion_message"
echo "========================================"
