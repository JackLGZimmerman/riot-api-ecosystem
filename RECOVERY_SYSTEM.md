# Recovery System

## Scope

Recovery in this pipeline is now intentionally basic:

- `players`: snapshot write + timestamp anchor, with rollback of partial run rows.
- `match_ids`: single-pass crawl + timestamp anchor, with rollback of run rows if save fails.
- `match_data`: durable queue using `game_data.matchdata_matchids` (row exists = pending; completed rows are deleted).

Prefect deployment concurrency is set to `1`, so only one pipeline run should execute at a time.

## Flow Modes

- Full mode runs `players` -> `match_ids` -> `match_data`; this is the default for `./restart.sh`.
- Matchdata-only mode runs only `match_data`; use `./restart.sh --matchdata-only` when the existing latest matchids run has enough regional coverage and the goal is to drain/fill match payloads.
- `./restart.sh --fresh --matchdata-only` is invalid because `--fresh` removes Docker volumes, including the ClickHouse data matchdata-only mode depends on.
- Matchdata-only restarts pause and do not resume `AUTOMATION_NAME`; a normal automation follow-up would use default flow parameters and collect matchids again unless separately configured for `matchdata_only=true`.
- If `AUTOMATION_NAME` is unset, pause any external full-pipeline automation before a matchdata-only restart.

## Active Recovery Components

| File | Responsibility |
|---|---|
| `app/worker/pipelines/prefect_flow.py` | Runs full or matchdata-only flow steps based on the Prefect `matchdata_only` parameter. |
| `app/worker/pipelines/players_orchestrator.py` | Deletes partial player rows on save failure, then writes/rotates players snapshot timestamp on success. |
| `app/worker/pipelines/matchids_orchestrator.py` | Writes `matchids` + successful player keys + timestamp; on failure deletes run rows and failed timestamp. |
| `app/worker/pipelines/matchdata_orchestrator.py` | Claims queue rows, writes match payloads, removes successful queue rows, keeps failed rows for retry. |
| `database/clickhouse/operations/work_state.py` | Matchdata queue operations only: seed from latest matchids run, claim rows, remove completed. |
| `database/clickhouse/operations/matchdata.py` | Delete helper for failed match subsets by `run_id` + `matchid`. |
| `database/clickhouse/operations/matchids.py` | Matchids anchor load/store + cleanup for failed/old runs. |

## Matchdata Queue Model

Queue table: `game_data.matchdata_matchids`

- `run_id`: source matchids run that discovered this match id.
- `matchid`: unit of work.
- Queue state is encoded by row existence:
- pending = row exists
- finished = row deleted

Flow:

1. On first loader call in a matchdata run, seed queue from existing `game_data.matchids`, excluding only matchids already present in both `info` and `tl_game_end` and matchids already in the queue.
2. Record a seed anchor (`data_timestamps.name = 'matchdata_seeded_available_matchids_run'`) tied to the latest `matchids_puuids_ts` run so the same available inventory is not reseeded on restart.
3. Claim next `MATCHDATA_CLAIM_BATCH_SIZE` pending matchids (per-continent round-robin).
4. Fetch non-timeline and timeline payloads concurrently.
5. Persist parsed rows.
6. Per-match resolution at end of batch:
   - Both streams succeeded: delete queue row, keep persisted rows.
   - One stream succeeded and the other returned terminal, or both streams returned terminal: delete partial persisted rows, delete the source `matchids` row, and delete the queue row.
   - Any retryable failure (5xx exhausted, retry pending): delete partial persisted rows and leave the queue row in place for the next batch.
7. Repeat until no pending rows remain.

Metadata-only gaps are not queue work. If `info` and `tl_game_end` both exist but
`metadata` is missing, do not full-requeue those matchids: full requeue causes the
collector to delete already-persisted raw rows and can trigger expensive
ClickHouse mutations on timeline tables. Use
`scripts/repair_partial_matchdata.py` instead; its default `auto` path classifies
metadata-only gaps and backfills only `game_data.metadata` from
`participant_stats.puuid` ordered by `participantid`, then clears repaired queue
rows after post-insert validation. True stream partials (`info`/`tl_game_end` XOR)
remain on the old backup-and-requeue path, but `--apply` refuses that path unless
`--allow-full-requeue` is passed.

## Simplifications Applied (2026-03-09)

- Removed obsolete pending-batch subsystem code from `work_state.py`.
- Removed the old orchestrator work-state schema file (no active callers).
- Recovery docs now describe only the active queue model.
- Matchdata seeding remains one-time per orchestrator run (`MatchDataLoader._initialized`), avoiding reseed query per batch.

## Operational Checks

Pending count:

```sql
SELECT count()
FROM game_data.matchdata_matchids;
```

Latest matchids anchor:

```sql
SELECT max(stored_at) AS ts, argMax(run_id, stored_at) AS run_id
FROM game_data.data_timestamps
WHERE name = 'matchids_puuids_ts';
```

## Known Constraint

`claim_pending_matchids` is a read claim, not a `processing` lock transition. With deployment concurrency `1`, this is acceptable. If parallel matchdata workers are introduced later, claim logic must move to an atomic claim model.
