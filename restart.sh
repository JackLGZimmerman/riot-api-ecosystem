#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export RIOT_PROJECT_DIR="$PROJECT_ROOT"

WORK_POOL_NAME="${WORK_POOL_NAME:-docker-pool}"
WORK_QUEUE_NAME="${WORK_QUEUE_NAME:-default}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-riot-pipeline/riot-pipeline}"
AUTOMATION_NAME="${AUTOMATION_NAME:-}"

fresh=0
matchdata_only=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --fresh)
      fresh=1
      ;;
    --matchdata-only)
      matchdata_only=1
      ;;
    *)
      echo "Usage: $0 [--fresh] [--matchdata-only]" >&2
      exit 2
      ;;
  esac
  shift
done

if [ "$fresh" -eq 1 ] && [ "$matchdata_only" -eq 1 ]; then
  echo "--fresh cannot be combined with --matchdata-only because it removes ClickHouse data." >&2
  exit 2
fi

if [ "$matchdata_only" -eq 1 ] && [ -z "$AUTOMATION_NAME" ]; then
  echo "Warning: --matchdata-only cannot pause unnamed Prefect automations; ensure any full-pipeline automation is already paused." >&2
fi

if [ ! -f "$RIOT_PROJECT_DIR/.env" ]; then
  echo "Missing $RIOT_PROJECT_DIR/.env; Prefect run containers need it mounted at /app/.env." >&2
  exit 1
fi

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

echo "Clearing stale Prefect runs"
PREFECT_API_URL="http://localhost:4200/api" \
  .venv/bin/python scripts/cancel_deployment_runs.py "$DEPLOYMENT_NAME"

if [ -n "$AUTOMATION_NAME" ] && [ "$matchdata_only" -eq 1 ]; then
  echo "Pausing automation for matchdata-only run: $AUTOMATION_NAME"
  PREFECT_API_URL="http://localhost:4200/api" \
    timeout 20s .venv/bin/prefect automation pause "$AUTOMATION_NAME"
fi

echo "Resuming Prefect intake"
PREFECT_API_URL="http://localhost:4200/api" \
  timeout 20s .venv/bin/prefect work-queue resume "$WORK_QUEUE_NAME" -p "$WORK_POOL_NAME"

if [ -n "$AUTOMATION_NAME" ] && [ "$matchdata_only" -eq 0 ]; then
  PREFECT_API_URL="http://localhost:4200/api" \
    timeout 20s .venv/bin/prefect automation resume "$AUTOMATION_NAME"
elif [ -n "$AUTOMATION_NAME" ]; then
  echo "Leaving automation paused for matchdata-only run: $AUTOMATION_NAME"
fi

echo "Starting new run"
run_args=()
if [ "$matchdata_only" -eq 1 ]; then
  run_args+=("--param" "matchdata_only=true")
fi
run_args+=("$DEPLOYMENT_NAME")
PREFECT_API_URL="http://localhost:4200/api" \
  .venv/bin/prefect deployment run "${run_args[@]}"
