#!/usr/bin/env bash

set -euo pipefail

PREFECT_API_URL="http://localhost:4200/api" \
  timeout 20s .venv/bin/prefect deployment schedule pause --all || true

mapfile -t flow_run_containers < <(
  docker ps -q --filter label=io.prefect.flow-run-id
)

for container_id in "${flow_run_containers[@]}"; do
  docker exec "$container_id" sh -c \
    'p="${PIPELINE_STOP_FLAG_PATH:-/tmp/riot_pipeline_stop_requested}"; mkdir -p "$(dirname "$p")" && : > "$p"' >/dev/null || true
done

if [ "${#flow_run_containers[@]}" -gt 0 ]; then
  docker wait "${flow_run_containers[@]}" >/dev/null 2>&1 || true
fi

docker stop prefect-worker
