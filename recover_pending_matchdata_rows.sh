#!/usr/bin/env bash

set -euo pipefail

mode="${1:---check}"

case "$mode" in
  --check|--apply)
    ;;
  -h|--help)
    echo "Usage: $0 [--check|--apply]"
    exit 0
    ;;
  *)
    echo "Usage: $0 [--check|--apply]" >&2
    exit 2
    ;;
esac

CLICKHOUSE_CONTAINER="${CLICKHOUSE_CONTAINER:-clickhouse}"

clickhouse() {
  docker exec "$CLICKHOUSE_CONTAINER" clickhouse-client \
    --connect_timeout=30 \
    --receive_timeout=3600 \
    --send_timeout=3600 \
    --query "$1"
}

pending_string_ids="
SELECT DISTINCT matchid
FROM game_data.matchdata_matchids
"

string_tables=(
  game_data.metadata
)

matchid_full_tables=(
  game_data.info
  game_data.bans
  game_data.feats
  game_data.objectives
  game_data.participant_stats
  game_data.participant_challenges
  game_data.participant_perk_values
  game_data.participant_perk_ids
  game_data.tl_participant_stats
  game_data.tl_building_kill
  game_data.tl_champion_kill
  game_data.tl_champion_special_kill
  game_data.tl_dragon_soul_given
  game_data.tl_elite_monster_kill
  game_data.tl_payload_event
  game_data.tl_turret_plate_destroyed
  game_data.tl_ck_victim_damage_dealt
  game_data.tl_ck_victim_damage_received
)

overlap_rows=0
pending_before="$(
  clickhouse "SELECT count() FROM game_data.matchdata_matchids FORMAT TabSeparatedRaw"
)"

show_overlap_rows() {
  local table
  local count
  local found=0

  overlap_rows=0

  for table in "${string_tables[@]}"; do
    count="$(
      clickhouse "
SELECT count()
FROM ${table}
WHERE matchid IN (${pending_string_ids})
FORMAT TabSeparatedRaw
"
    )"
    overlap_rows=$((overlap_rows + count))
    if [ "$count" != "0" ]; then
      printf '%s %s\n' "$table" "$count"
      found=1
    fi
  done

  for table in "${matchid_full_tables[@]}"; do
    count="$(
      clickhouse "
SELECT count()
FROM ${table}
WHERE matchidfull IN (${pending_string_ids})
FORMAT TabSeparatedRaw
"
    )"
    overlap_rows=$((overlap_rows + count))
    if [ "$count" != "0" ]; then
      printf '%s %s\n' "$table" "$count"
      found=1
    fi
  done

  if [ "$found" -eq 0 ]; then
    echo "No overlapping rows found."
  fi

  echo "Total overlapping rows: $overlap_rows"
}

delete_overlap_rows() {
  local table

  for table in "${string_tables[@]}"; do
    echo "Deleting $table"
    clickhouse "
ALTER TABLE ${table}
DELETE
WHERE matchid IN (${pending_string_ids})
SETTINGS mutations_sync = 2
"
  done

  for table in "${matchid_full_tables[@]}"; do
    echo "Deleting $table"
    clickhouse "
ALTER TABLE ${table}
DELETE
WHERE matchidfull IN (${pending_string_ids})
SETTINGS mutations_sync = 2
"
  done
}

echo "Pending queue rows: $pending_before"
show_overlap_rows

if [ "$mode" = "--check" ]; then
  if [ "$overlap_rows" -eq 0 ]; then
    echo "No partial rows detected for currently pending matchids."
    exit 0
  fi

  echo "Partial rows detected for currently pending matchids." >&2
  exit 1
fi

if [ "$overlap_rows" -eq 0 ]; then
  echo "Nothing to delete. Queue left unchanged."
  exit 0
fi

delete_overlap_rows

pending_after="$(
  clickhouse "SELECT count() FROM game_data.matchdata_matchids FORMAT TabSeparatedRaw"
)"

echo "Pending queue rows after cleanup: $pending_after"
show_overlap_rows

if [ "$pending_before" != "$pending_after" ]; then
  echo "Pending queue size changed from $pending_before to $pending_after." >&2
  exit 1
fi

if [ "$overlap_rows" -ne 0 ]; then
  echo "Pending matchids still have rows in matchdata tables." >&2
  exit 1
fi

echo "Deleted partial rows for pending matchids. Queue rows were left in place."
