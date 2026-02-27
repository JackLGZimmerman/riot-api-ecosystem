#!/usr/bin/env bash

set -euo pipefail

echo "========================================"
echo "Restart: stopping containers (preserving volumes)..."
echo "========================================"
docker compose down --remove-orphans

echo "========================================"
echo "Cleaning old Prefect flow-run containers..."
echo "========================================"
docker ps -aq --filter label=io.prefect.flow-run-id | xargs -r docker rm -f

echo "========================================"
echo "Starting containers with rebuild and waiting for health..."
echo "========================================"
docker compose up -d --build --wait

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
echo "Restart complete"
echo "========================================"
