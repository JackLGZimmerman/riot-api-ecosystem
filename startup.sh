#!/usr/bin/env bash

set -euo pipefail

echo "========================================"
echo "Stopping and removing containers..."
echo "========================================"
docker compose down -v

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
echo "Deployment complete ✅"
echo "========================================"
