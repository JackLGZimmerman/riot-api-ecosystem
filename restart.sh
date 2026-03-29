#!/usr/bin/env bash

set -euo pipefail

WORK_POOL_NAME="${WORK_POOL_NAME:-docker-pool}"
WORK_QUEUE_NAME="${WORK_QUEUE_NAME:-default}"
AUTOMATION_NAME="${AUTOMATION_NAME:-}"

case "${1:-}" in
  "")
    fresh=0
    ;;
  --fresh)
    fresh=1
    ;;
  *)
    echo "Usage: $0 [--fresh]" >&2
    exit 2
    ;;
esac

for path in \
  app/core/logging/logs/app.log.jsonl \
  app/core/logging/logs/schema_drift/non_timeline.log.jsonl \
  app/core/logging/logs/schema_drift/timeline.log.jsonl
do
  mkdir -p "$(dirname "$path")"
  rm -f "$path"
  touch "$path"
  chmod u+rw "$path"
done

echo "Stopping current pipeline"
./stop_pipeline_safely.sh || true

echo "Resetting containers"
if [ "$fresh" -eq 1 ]; then
  docker compose down -v --remove-orphans
else
  docker compose down --remove-orphans
fi

docker ps -aq --filter label=io.prefect.flow-run-id | xargs -r docker rm -f

echo "Building flow image"
docker build --no-cache -t riot-pipeline:latest .

echo "Starting services"
docker compose up -d --build --wait

echo "Cancelling old Prefect runs"
docker exec prefect-server python - <<'PY'
import requests
import time

base = "http://localhost:4200/api"
deployment_name = "riot-pipeline/riot-pipeline"
message = "Cancelled during restart cleanup before launching a new deployment run."

response = requests.get(f"{base}/deployments/name/{deployment_name}", timeout=30)
if response.status_code == 404:
    raise SystemExit(0)
response.raise_for_status()
deployment_id = response.json()["id"]

def fetch_runs():
    response = requests.post(
        f"{base}/flow_runs/filter",
        json={"flow_runs": {"state": {"type": {"any_": ["RUNNING", "PENDING", "SCHEDULED", "CANCELLING"]}}}},
        timeout=60,
    )
    response.raise_for_status()
    return [run for run in response.json() if run.get("deployment_id") == deployment_id]

def active_slots():
    response = requests.get(f"{base}/deployments/{deployment_id}", timeout=30)
    response.raise_for_status()
    return ((response.json().get("global_concurrency_limit") or {}).get("active_slots") or 0)

for _ in range(30):
    runs = fetch_runs()
    for run in runs:
        cancel = requests.post(
            f"{base}/flow_runs/{run['id']}/set_state",
            json={
                "state": {
                    "type": "CANCELLED",
                    "name": "Cancelled",
                    "message": message,
                },
                "force": True,
            },
            timeout=60,
        )
        cancel.raise_for_status()
        if cancel.json().get("status") != "ACCEPT":
            raise SystemExit(f"Failed to cancel {run['id']}: {cancel.text}")

    if not fetch_runs() and active_slots() == 0:
        raise SystemExit(0)

    time.sleep(2)

raise SystemExit("Non-terminal Prefect runs remain after cleanup.")
PY

echo "Deploying flow"
.venv/bin/prefect --no-prompt deploy --prefect-file prefect.yaml

echo "Resuming Prefect intake"
PREFECT_API_URL="http://localhost:4200/api" \
  timeout 20s .venv/bin/prefect work-queue resume "$WORK_QUEUE_NAME" -p "$WORK_POOL_NAME" || true

if [ -n "$AUTOMATION_NAME" ]; then
  PREFECT_API_URL="http://localhost:4200/api" \
    timeout 20s .venv/bin/prefect automation resume "$AUTOMATION_NAME" || true
fi

echo "Starting new run"
.venv/bin/prefect deployment run 'riot-pipeline/riot-pipeline'
