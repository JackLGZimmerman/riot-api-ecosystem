#!/usr/bin/env bash

set -euo pipefail

WORK_POOL_NAME="${WORK_POOL_NAME:-docker-pool}"
WORK_QUEUE_NAME="${WORK_QUEUE_NAME:-default}"
AUTOMATION_NAME="${AUTOMATION_NAME:-}"
CLICKHOUSE_CONTAINER="${CLICKHOUSE_CONTAINER:-clickhouse}"
STOP_WAIT_TIMEOUT_S="${STOP_WAIT_TIMEOUT_S:-600}"

PREFECT_API_URL="http://localhost:4200/api" \
  timeout 20s .venv/bin/prefect work-queue pause "$WORK_QUEUE_NAME" -p "$WORK_POOL_NAME" || true

if [ -n "$AUTOMATION_NAME" ]; then
  PREFECT_API_URL="http://localhost:4200/api" \
    timeout 20s .venv/bin/prefect automation pause "$AUTOMATION_NAME" || true
fi

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

unfinished_matchdata_mutations() {
  docker exec "$CLICKHOUSE_CONTAINER" clickhouse-client \
    --connect_timeout=30 \
    --receive_timeout=3600 \
    --send_timeout=3600 \
    --query "
SELECT count()
FROM system.mutations
WHERE database = 'game_data'
  AND is_done = 0
  AND (
    table = 'matchdata_matchids'
    OR table IN (
      'metadata',
      'info',
      'bans',
      'feats',
      'objectives',
      'participant_stats',
      'participant_challenges',
      'participant_perk_values',
      'participant_perk_ids'
    )
    OR startsWith(table, 'tl_')
  )
FORMAT TabSeparatedRaw
"
}

echo "Waiting for ClickHouse matchdata mutations to finish"
deadline=$((SECONDS + STOP_WAIT_TIMEOUT_S))
while true; do
  remaining="$(unfinished_matchdata_mutations)"
  if [ "$remaining" = "0" ]; then
    break
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "Timed out waiting for ClickHouse matchdata mutations to finish ($remaining remaining)." >&2
    exit 1
  fi
  sleep 2
done
