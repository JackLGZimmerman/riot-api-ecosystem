# HGNN Central Band Review

Date: 2026-06-04

Status note, 2026-06-10: this is a historical methodology note, not a current
production decision record. The review should be repeated against newer
production models after the chronological split boundary is rolled forward. For
current experiment rules and the next data direction, see
[EXPERIMENTS.md](EXPERIMENTS.md). For the current model surface, see
[HGNN_CURRENT.md](HGNN_CURRENT.md).

## Premise

The original review inspected held-out validation/test games where the reference
HGNN predicted `P(blue win)` near the decision boundary and the raw `0.5`
classification was wrong. The goal was to identify allowed, pregame-safe signal
families that could plausibly explain central-band misses without using player
identity or completed-game leakage.

Two bands were sampled:

- Tight central band: `P(blue win) in [0.475, 0.525]`.
- Wider diagnostic band: `P(blue win) in [0.425, 0.575]`.

Allowed evidence was restricted to champion, role/position, build label,
historical champion/build/relationship performance, summoner spells, runes, and
patch/date. No player identity evidence was selected, emitted, aggregated, or
used for reasoning. Rune joins used `puuid` only inside ClickHouse predicates to
align rune rows to participant slots; the analyzer did not materialize player
identity.

## Methodology

Rebuild the review rather than relying on the historical outputs:

1. Generate central-band miss candidates from the current production checkpoint
   and current cache.
2. Sample misses in reproducible batches from the tight band first; use the
   wider band only to test whether the same signal families persist outside the
   sharpest boundary slice.
3. For each sampled game, compare the model prediction with split-safe,
   train-derived evidence only:
   - champion/build 1vX priors and support,
   - allowed direct relationship or matchup diagnostics when intentionally
     included for forensics,
   - summoner-spell and rune/page priors,
   - patch/date cohorts,
   - draft-safe loadout inputs.
4. Keep completed-game build-profile labels, secondary build margins, and item
   value totals as oracle diagnostics only unless a draft-safe build-intent
   source supplies them before the game outcome.
5. Report counts as overlapping influence families, because one game can expose
   more than one missing signal.
6. Validate any proposed signal with the experiment gates in
   [EXPERIMENTS.md](EXPERIMENTS.md): central-band NLL lift is required, and
   accuracy-only movement is not promotion evidence.

## Historical Artifacts

The retired 2026-06-04 review produced candidate batches, allowed-signal
analyses, wider-band analyses, and lift-estimate JSON files under
`app/ml/data/experiments/`. The one-off generator/analyzer scripts are no longer
part of the maintained workspace. Treat those artifacts as provenance for the
old forensic pass, not as current evaluation fixtures.

## Repeat Criteria

Repeat this review only after a meaningful production-model or data-boundary
change. The current next step is the data refresh/rolled chronological split
protocol in [EXPERIMENTS.md](EXPERIMENTS.md#next-data-direction), because the
2026-06-10 ceiling work localized the remaining boundary headroom to same-patch
history and split freshness rather than another semantic architecture sweep.
