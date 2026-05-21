# Match-Level MoE Evaluation

The MoE head has been removed from the model and training code. This file
records why, and the conditions a future MoE attempt must meet.

## Why It Was Removed

A match-level MoE head was implemented, swept (2026-05-19), and never beat the
dense antisymmetric head. The dense head won on validation loss with no
calibration regression, so `use_moe=false` was the default; the unused code was
then removed to keep the model lean.

The router was degenerate: full-router entropy `2.0794` (`ln 8`), top-k margin
`0.0005`. The MoE acted as a monotonic confidence stretch — pushing low
probabilities lower and high higher — not as a distinct decision surface. It
did not improve the central 40-60% band on BCE, Brier, AUC, or hard accuracy,
and worsened high-side ECE.

## Why It Failed (Structural, Not Tuning)

- The head sits downstream of team pooling. It saw only the pooled 6d
  `match_features` plus three `baseline_logit` scalars — the same view the
  dense head has. An MoE head partitions a decision surface; it cannot recover
  signal the trunk discarded in pooling.
- Router input equalled expert input. Every expert saw the same pooled vector,
  so there was no axis to specialize on.
- The Switch load-balancing aux loss forces uniform 8-way usage. With identical
  inputs, a uniform router is the only solution — the objective manufactures
  the degeneracy.

No hyperparameter on identical-input experts produces decisive routing.

## Conditions For A Future MoE

An MoE earns its place only if the feature→outcome map is genuinely piecewise
across regimes, and the router can see the regime. To attempt it again:

1. Confirm the temporal signal survives pooling first. Check the trained
   `profile_gate` and whether `delta_proj` moved off zero-init. If `team_mean`
   averages the per-player delta into noise, fix pooling — no head change helps.
2. Give the router a real specialization axis: an oriented team-tempo summary
   built from the per-team temporal `delta` (scaling vs early-tempo comps),
   computed before pooling collapses it. It must be oriented (blue-relative,
   flipped for `m(r,b)`) to keep the `score_bvr - score_rvb` antisymmetry.
3. Drop to 2-4 experts and anneal the aux loss toward 0 after warmup, so a
   router with a real axis is free to be decisive.
4. Prefer the simpler control first: concatenate the team-tempo summary into
   the dense head input. Tempo is likely a smooth modulation, not a piecewise
   split, so dense conditioning probably captures most of the gain with none of
   the routing instability. Reach for MoE only if conditioning shows evidence
   of a genuine regime discontinuity.

## Promotion Gate

Gate on central-band hard accuracy and central/tail ECE separately — not
headline loss or confidence spread. Spreading probabilities away from 50% is
overconfidence inflation, not uncertainty reduction. A real win moves
central-band games to the correct side: central hard accuracy up, central and
tail ECE flat or better. Require a repeatable validation-loss win plus no
calibration regression.

## Research Anchors

- [Shazeer et al. 2017](https://arxiv.org/abs/1701.06538): sparse MoE relies on learned routing and conditional capacity.
- [Switch Transformer](https://arxiv.org/abs/2101.03961) and [ST-MoE](https://arxiv.org/abs/2202.08906): routing stability and load management are core design concerns.
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368): route balance affects training quality, not just efficiency.
