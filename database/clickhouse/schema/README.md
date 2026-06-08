# ClickHouse Schema Conventions

Bootstrap DDL for the raw ingestion DB. Files are mounted read-only into
`/docker-entrypoint-initdb.d` and run **only on a fresh ClickHouse data dir**
(every statement is `CREATE TABLE IF NOT EXISTS`). Editing a file does **not**
alter an existing table — apply such changes to a live DB with a manual
migration.

Numbering: `NNNN_<name>_schema.sql`. Schema and build files share the number
prefix. `0xxx` bootstrap, `1xxx` players, `2xxx` matchids, `3xxx` raw match data
(`31xx` non-timeline, `31xx`/`3100+` timeline `tl_*`). `4xxx+` are downstream
/derived tables (out of scope for the ingestion pipeline).

## Column conventions (enforced going forward)

- `run_id UUID` first column on every ingested table.
- `matchid String CODEC (ZSTD(3))`, `puuid FixedString(78) CODEC (ZSTD(3))` —
  the high-cardinality id columns always carry the ZSTD(3) codec.
- Timeline tables: `frame_timestamp UInt32` (frame index, **not** a time),
  `timestamp UInt64` (event epoch ms). Other ms epochs (`realtimestamp`,
  `actualstarttime`) are `UInt64`; second epochs (`stored_at`) are `UInt32`.
- Actor ids: `UInt8` when always present (`participantid`, `creatorid`,
  `teamid`); `Int8` when a `-1` sentinel is possible (`killerid`, `victimid`);
  `killerteamid` is `Int16` for its non-team sentinel.
- Event discriminator `type` is `LowCardinality(String)`; enums use `Enum8`
  (canonical casing, not `ENUM8`).

## Engine / ORDER BY / partitioning (D1–D4)

Structural choices — changing them on a live table is a full rebuild + migration.
Resolved as of 2026-06-08:

- **Engine (D1) — verified, no change.** `ReplacingMergeTree` for tables keyed for
  idempotent re-ingest (`1001_players`, `2000_matchid_puuids`,
  `3000_matchdata_matchids`); `MergeTree` for append-only event tables (all other
  `3xxx`, `0001`, `2001`).
- **Dedup grain (D2) — verified, no change.** The three Replacing tables encode
  three intentional grains:
  - `3000_matchdata_matchids` `ORDER BY (matchid)` — dedups across runs.
  - `1001_players` `ORDER BY (puuid, queue_type, region, updated_at, run_id)` —
    keeps versioned rows (run_id last).
  - `2000_matchid_puuids` `ORDER BY (run_id, puuid, queue_type)` — `run_id`-leading
    is correct: this is a **per-run snapshot**. The ingest layer reads only the
    latest run (`load_matchid_puuids` → `WHERE run_id = argMax(run_id, stored_at)`)
    and deletes old runs explicitly; cross-run dedup is deliberately not wanted.
- **Partitioning (D3) — decided unpartitioned.** Only `1001_players` is partitioned
  (`PARTITION BY toDate(updated_at)`). The high-volume `3xxx` tables stay
  unpartitioned: most `tl_*` tables carry no date column (only `matchid` +
  `frame_timestamp`), merges are matchid-keyed, and there is no retention /
  drop-by-date use case to justify the extra merge overhead.
- **Sort keys (D4) — applied.** `tl_*` ORDER BY keys trimmed to
  `(matchid, frame_timestamp, timestamp[, primary actor id])`, dropping trailing
  low-selectivity value columns (e.g. `level` in `tl_level_up`). Append-only
  `MergeTree`, so row counts are unaffected. Live migration:
  `migrations/2026-06-08_d4_trim_tl_sort_keys.sh` (run with the pipeline stopped).
  Tables whose trailing column is a high-selectivity row/event id are unchanged
  (`tl_participant_stats`, `tl_champion_kill`, `tl_ck_victim_damage_*`,
  `tl_objective_bounty_finish`).
