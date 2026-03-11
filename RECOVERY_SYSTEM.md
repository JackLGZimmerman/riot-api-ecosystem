# Recovery System

## Scope

Recovery in this pipeline is now intentionally basic:

- `players`: snapshot write + timestamp anchor, with rollback of partial run rows.
- `match_ids`: single-pass crawl + timestamp anchor, with rollback of run rows if save fails.
- `match_data`: durable per-match queue using `game_data.matchdata_matchids` (`pending` / `finished`).

Prefect deployment concurrency is set to `1`, so only one full pipeline run should execute at a time.

## Active Recovery Components

| File | Responsibility |
|---|---|
| `app/worker/pipelines/prefect_flow.py` | Runs `players` -> `match_ids` -> `match_data` sequentially per flow run. |
| `app/worker/pipelines/players_orchestrator.py` | Deletes partial player rows on save failure, then writes/rotates players snapshot timestamp on success. |
| `app/worker/pipelines/matchids_orchestrator.py` | Writes `matchids` + successful player keys + timestamp; on failure deletes run rows and failed timestamp. |
| `app/worker/pipelines/matchdata_orchestrator.py` | Claims pending matchids, writes match payloads, marks success `finished`, requeues failed items as `pending`. |
| `database/clickhouse/operations/work_state.py` | Matchdata queue operations only: schema ensure, seed from latest matchids run, claim pending, mark pending/finished. |
| `database/clickhouse/operations/matchdata.py` | Delete helper for failed match subsets by `run_id` + `matchid`. |
| `database/clickhouse/operations/matchids.py` | Matchids anchor load/store + cleanup for failed/old runs. |

## Matchdata Queue Model

Queue table: `game_data.matchdata_matchids`

- `run_id`: source matchids run that discovered this match id.
- `matchid`: unit of work.
- `status`: `pending` or `finished`.
- `last_error`: last retry reason for pending rows.

Flow:

1. On first loader call in a matchdata run, seed queue from latest `matchids` run (`data_timestamps.name = 'matchids_puuids_ts'`), only inserting unseen `matchid`.
2. Claim next `MATCHDATA_BATCH_SIZE` pending matchids (`ORDER BY matchid`).
3. Fetch non-timeline and timeline payloads concurrently.
4. Persist parsed rows.
5. For per-match failures: delete inserted rows for failed matches and set queue status back to `pending` with `last_error`.
6. For successful matches: set queue status to `finished`.
7. Repeat until no pending rows remain.

## Simplifications Applied (2026-03-09)

- Removed obsolete pending-batch subsystem code from `work_state.py`.
- Removed obsolete schema file `database/clickhouse/schema/3001_orchestrator_work_state.sql` (no active callers).
- Recovery docs now describe only the active status-queue model.
- Matchdata seeding remains one-time per orchestrator run (`MatchDataLoader._initialized`), avoiding reseed query per batch.

## Operational Checks

Pending/finished counts:

```sql
SELECT status, count()
FROM game_data.matchdata_matchids
GROUP BY status
ORDER BY status;
```

Most common pending errors:

```sql
SELECT last_error, count()
FROM game_data.matchdata_matchids
WHERE status = 'pending'
GROUP BY last_error
ORDER BY count() DESC
LIMIT 20;
```

Latest matchids anchor:

```sql
SELECT max(stored_at) AS ts, argMax(run_id, stored_at) AS run_id
FROM game_data.data_timestamps
WHERE name = 'matchids_puuids_ts';
```

## Known Constraint

`claim_pending_matchids` is a read claim, not a `processing` lock transition. With deployment concurrency `1`, this is acceptable. If parallel matchdata workers are introduced later, claim logic must move to an atomic claim model.
