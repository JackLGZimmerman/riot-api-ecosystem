#!/usr/bin/env bash

set -euo pipefail

echo "========================================"
echo "Fresh start: stopping containers and removing volumes..."
echo "========================================"
docker compose down -v --remove-orphans

echo "========================================"
echo "Cleaning old Prefect flow-run containers..."
echo "========================================"
docker ps -aq --filter label=io.prefect.flow-run-id | xargs -r docker rm -f || true

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
prefect --no-prompt deploy --prefect-file prefect.yaml

echo "========================================"
echo "Running Prefect deployment..."
echo "========================================"
prefect deployment run 'riot-pipeline/riot-pipeline'

echo "========================================"
echo "Fresh start complete"
echo "========================================"
