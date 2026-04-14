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

echo "Stopping current pipeline"
./stop_pipeline_safely.sh

echo "Resetting containers"
if [ "$fresh" -eq 1 ]; then
  docker compose down -v --remove-orphans
else
  docker compose down --remove-orphans
fi

echo "Building flow image"
docker build -t riot-pipeline:latest .

echo "Starting services"
docker compose up -d --build --wait

echo "Deploying flow"
.venv/bin/prefect --no-prompt deploy --prefect-file prefect.yaml

echo "Resuming Prefect intake"
PREFECT_API_URL="http://localhost:4200/api" \
  timeout 20s .venv/bin/prefect work-queue resume "$WORK_QUEUE_NAME" -p "$WORK_POOL_NAME"

if [ -n "$AUTOMATION_NAME" ]; then
  PREFECT_API_URL="http://localhost:4200/api" \
    timeout 20s .venv/bin/prefect automation resume "$AUTOMATION_NAME"
fi

echo "Starting new run"
.venv/bin/prefect deployment run 'riot-pipeline/riot-pipeline'
