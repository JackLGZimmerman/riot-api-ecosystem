#!/usr/bin/env bash
#
# D4 migration: trim trailing low-selectivity value columns from tl_* ORDER BY keys.
# See database/clickhouse/schema/README.md (D4). Append-only MergeTree tables, so the
# sort key affects only the primary index/compression — row counts are preserved.
#
# Prereq: ingestion pipeline stopped (./stop_pipeline_safely.sh) so no writes are in
# flight. Run once against the live DB. Per table: build a shadow with the new ORDER BY,
# copy all rows, verify row-count parity, then atomically EXCHANGE. Old data lands in
# <t>__new after the swap and is only dropped when DROP_OLD=1.

set -euo pipefail

CLICKHOUSE_CONTAINER="${CLICKHOUSE_CONTAINER:-clickhouse}"
DB="${DB:-game_data}"
DROP_OLD="${DROP_OLD:-0}"   # set to 1 to drop the old (post-swap <t>__new) tables

# table -> new ORDER BY (must match the matching 3xxx_*_schema.sql exactly)
TABLES=(
  "tl_building_kill|matchid, frame_timestamp, timestamp, killerid"
  "tl_champion_special_kill|matchid, frame_timestamp, timestamp, killerid"
  "tl_dragon_soul_given|matchid, frame_timestamp, timestamp, teamid"
  "tl_elite_monster_kill|matchid, frame_timestamp, timestamp, killerid"
  "tl_turret_plate_destroyed|matchid, frame_timestamp, timestamp, killerid"
  "tl_ward_placed|matchid, frame_timestamp, timestamp, creatorid"
  "tl_ward_kill|matchid, frame_timestamp, timestamp, killerid"
  "tl_item_purchased|matchid, frame_timestamp, timestamp, participantid"
  "tl_item_sold|matchid, frame_timestamp, timestamp, participantid"
  "tl_item_destroyed|matchid, frame_timestamp, timestamp, participantid"
  "tl_item_undo|matchid, frame_timestamp, timestamp, participantid"
  "tl_level_up|matchid, frame_timestamp, timestamp, participantid"
  "tl_skill_level_up|matchid, frame_timestamp, timestamp, participantid"
  "tl_pause_end|matchid, frame_timestamp, timestamp"
  "tl_game_end|matchid, frame_timestamp, timestamp"
  "tl_objective_bounty_prestart|matchid, frame_timestamp, timestamp, teamid"
  "tl_feat_update|matchid, frame_timestamp, timestamp, teamid"
  "tl_champion_transform|matchid, frame_timestamp, timestamp, participantid"
)

ch() {
  docker exec "$CLICKHOUSE_CONTAINER" clickhouse-client \
    --connect_timeout=30 --receive_timeout=7200 --send_timeout=7200 \
    --query "$1"
}

count() { ch "SELECT count() FROM ${DB}.$1"; }

for entry in "${TABLES[@]}"; do
  t="${entry%%|*}"
  order_by="${entry#*|}"
  new="${t}__new"

  echo "=== ${t} -> ORDER BY (${order_by}) ==="

  ch "DROP TABLE IF EXISTS ${DB}.${new}"
  ch "CREATE TABLE ${DB}.${new} AS ${DB}.${t} ENGINE = MergeTree ORDER BY (${order_by})"

  echo "  copying rows..."
  ch "INSERT INTO ${DB}.${new} SELECT * FROM ${DB}.${t}"

  src="$(count "${t}")"
  dst="$(count "${new}")"
  if [ "$src" != "$dst" ]; then
    echo "  ROW COUNT MISMATCH src=${src} dst=${dst}; aborting (dropping ${new})" >&2
    ch "DROP TABLE ${DB}.${new}"
    exit 1
  fi
  echo "  row-count parity ok (${src})"

  ch "EXCHANGE TABLES ${DB}.${t} AND ${DB}.${new}"
  echo "  swapped"

  if [ "$DROP_OLD" = "1" ]; then
    ch "DROP TABLE ${DB}.${new}"
    echo "  dropped old ${new}"
  else
    echo "  kept old data as ${DB}.${new} (set DROP_OLD=1 to reclaim space)"
  fi
done

echo "Done. Verify sorting keys:"
echo "  SELECT name, sorting_key FROM system.tables WHERE database='${DB}' AND name LIKE 'tl_%';"
