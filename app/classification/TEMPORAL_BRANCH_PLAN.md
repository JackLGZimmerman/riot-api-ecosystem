# Temporal Branch Plan

Separate workstream from the full-game metric registry
([METRIC_CATALOGUE_PLAN.md](METRIC_CATALOGUE_PLAN.md)). Shares no code with it
beyond the word "metric"; feeds its own autoencoder.

## Goal

Build identity-level temporal tensors shaped `(identity, minute_bucket, metric)`
and a dedicated autoencoder over them. Independent of the full-game pipeline.

## Shape (built)

- Minute buckets `0..45` plus one `46_plus` overflow bucket (47 buckets);
  `bucket = least(intDiv(frame_timestamp, 60000), 46)`.
- 51 metrics = 45 `tl_participant_stats` gameplay stats + 6 events. Excluded:
  identifiers (`run_id`/`matchid`/`frame_timestamp`/`participantid`) and
  side-mirrored `position_x`/`position_y`.
- Events (`EVENT_METRICS`): `kills`/`assists`/`deaths` from `tl_champion_kill`
  (killer / assisting / victim participant) and `plate_top`/`plate_mid`/
  `plate_bot` from `tl_turret_plate_destroyed` (killer, by `lanetype`). Building
  and elite-monster kills are intentionally excluded as too sparse.
- One frame-count denominator for both families: the bucket's frame count is the
  number of the identity's games that reached that minute (one frame per
  game-minute), i.e. the game-end normaliser. Stat bucket value = `SUM / frames`;
  event bucket rate = `SUM / frames`.
- Per-`(identity, minute_bucket, metric)` frame-count shrinkage (the per-minute
  prior): identity cell -> champion-role bucket mean -> global bucket mean.
  Unobserved cells fall back to the parent. Mask marks observed buckets. Events
  ride the same prior, so per-minute event rates are shrunk identically.

## Implementation

- `game_data_filtered.temporal_identity_bins` — materialised table, one row per
  `(split, championid, teamposition, build, bucket)` carrying `frames` plus a
  SUM per metric (45 `sum_*` + 6 `ev_*`). Built server-side by
  `build_tables.build_temporal_table()`: the heavy 352M-frame scan is sharded by
  `cityHash64(matchid) % 8` into `temporal_stat_stage`, events into
  `temporal_ev_stage`, then `GROUP BY`-combined into the bins table (the
  `shard -> stage -> combine` pattern keeps each INSERT under the memory limit).
- `app/classification/embeddings/temporal.py` — reads the prepared table
  (`_load_bins`, a few hundred K rows), 2-level shrinkage, median/MAD
  standardisation, `(n, 47, 51)` tensor + mask, cached to
  `_raw/<split>_temporal.npz`.
- `app/classification/temporal_autoencoder.py` — fresh, compact mask-aware
  autoencoder (shared per-bucket metric embedding + champion/role/build
  embeddings + MLP). Reconstruction MSE skips unobserved buckets.

## Deferred

- Building / elite-monster kill events (too sparse to bin per minute).
- Documented derived TLPS ratios (e.g. damage-type splits) on top of the raw
  51-metric tensor.
