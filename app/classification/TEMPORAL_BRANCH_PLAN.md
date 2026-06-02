# Temporal Branch Plan

Separate workstream from the full-game metric registry
([METRIC_CATALOGUE_PLAN.md](METRIC_CATALOGUE_PLAN.md)). Shares no code with it
beyond the word "metric"; feeds its own autoencoder.

## Goal

Build identity-level temporal tensors shaped `(identity, minute_bucket, metric)`
and a dedicated autoencoder over them. Independent of the full-game pipeline.

## Shape

- Minute buckets `0..45` plus one `46_plus` overflow bucket.
- Metrics: numeric `tl_participant_stats` gameplay columns (except
  identifiers/payload), documented derived TLPS ratios, and key event metrics
  from `tl_champion_kill`, `tl_building_kill`, `tl_elite_monster_kill`.
- Per-`(identity, minute_bucket, metric)` priors shrink using participant-minute
  frame counts (`participant_minute_frames` evidence).

## Status

Not started. Deferred until full-game Phases 1–3 land, so the registry's
`evidence_kind` abstraction is proven before extending it to a third evidence
class.

## Open items

- Confirm the timeline catalogues (72 TLPS metrics, 117 event metrics) against
  the live schema before sizing the tensor.
- Masking strategy for games shorter than a given bucket.
- Whether the temporal autoencoder reuses `champion_semantics.py` building
  blocks or is a fresh module.
