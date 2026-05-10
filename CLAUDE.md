# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Philosophy

### Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.(())

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
- The test: Every changed line should trace directly to the user's request.

### Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

```bash
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### Documentation (.md)

Documentation is to be written in `.md` files.

- All documentation should be kept as lean as possible, no superflous text, straight to the point, minimal details and maximum clarity.

## Architecture

### High-Level Data Flow

```
Riot API → [Players → Match IDs → Match Data] → ClickHouse → Filtered Tables → ML Model
                  (Prefect orchestrated, concurrency=1)
```

The pipeline has three sequential stages, each with its own orchestrator:

| Stage | Orchestrator | Output Table |
|---|---|---|
| 1. Players | `app/worker/pipelines/players_orchestrator.py` | `game_data.players` |
| 2. Match IDs | `app/worker/pipelines/matchids_orchestrator.py` | `game_data.matchids` |
| 3. Match Data | `app/worker/pipelines/matchdata_orchestrator.py` | `game_data.info`, `game_data.participant_stats`, `game_data.tl_*` |

Entry point: `app/worker/pipelines/prefect_flow.py:riot_pipeline`

### Key Modules

- **`app/services/riot_api_client/`** — async Riot API client with rate limiting (100 calls/120s) and exponential backoff retry; separate parsers for non-timeline and timeline payloads (Pydantic models)
- **`app/core/config/settings.py`** — Pydantic BaseSettings loaded from `.env` (API key, ClickHouse creds, rate limit params)
- **`app/core/config/constants/`** — geography (regions/continents), endpoints, queue type parameters
- **`database/clickhouse/client.py`** — connection pooling to ClickHouse
- **`database/clickhouse/operations/`** — all DB read/write operations
- **`app/ml/`** — TransformerSetModel (permutation-invariant over 10 participant tokens) for blue-team win prediction; see `app/ml/README.md`

### ClickHouse Schema

SQL files are numbered by stage. Schema (table DDL) and build (INSERT/materialized view) files share the same prefix number — see `database/clickhouse/schema/`.

| Range | Purpose |
|---|---|
| 3xxx | Raw ingestion tables (`participant_stats`, corrections) |
| 4xxx | Filter tables and bitmask logic |
| 5xxx | Derived/aggregated analytical tables |
| 7xxx | Dictionary tables (item image map) |
| 8xxx | Ad-hoc analytics queries (`analytics_builds/`) |

Two databases: `game_data` (raw ingestion) and `game_data_filtered` (post-filter analytical tables).

For filter/analytics rebuild order and ClickHouse commands, see `database/clickhouse/commands.md`.

For filter rule semantics and thresholds, see `database/clickhouse/filter_evidence.md`.

### Recovery System

Match data uses a durable queue: row presence in `game_data.matchdata_matchids` signals pending work. Players and match IDs use timestamp anchors with partial-run rollback. See `RECOVERY_SYSTEM.md` for full details.

### Infrastructure

Docker Compose services: ClickHouse (ports 8123/9000), Prefect Server (port 4200) + PostgreSQL backend, Prefect Worker (docker-pool). Deployment concurrency is 1 with CANCEL_NEW strategy — a new run requested during an active run is dropped, not queued.

## Applied Learning

On multiple failures, work-arounds, or limitations, add a bullet point that will save time in future sessions. Keep each bullet under 15 words.

- SQLFluff dialect is ClickHouse; max line length 88 (matches `.sqlfluff` config).
