# HGNN Context Examples Audit

Updated: 2026-06-10.

Status note: keep this document as a qualitative evaluation fixture. The
specific group-context instances are useful for inspecting semantic failures,
but promotion decisions should use the lower-noise EB/group guardrail in
[HGNN_GROUP_CONTEXT_AUDIT.md](HGNN_GROUP_CONTEXT_AUDIT.md) plus the NLL-first
rules in [EXPERIMENTS.md](EXPERIMENTS.md).

This fixture keeps only examples whose populated threshold bins had `n >= 500`
in the last rendered validation audit. Low-support examples were removed because
their tail bins can dominate `Gap`, `Gap MSE`, and especially `Max Abs Gap`
without enough sample size to make those values stable. Regenerate the audit on
the refreshed cache before treating the retained support counts as current.
Max-gap readings should still be paired with bin `n`; p95 or EB/group metrics
are better headline guardrails.

## Scope And Threshold Definitions

- Context source: `app/ml/data/cache` side-row arrays.
- HGNN model: `app/ml/data/hgnn_production_model.pt`.
- Encoder sidecar artifact: `app/ml/data/semantic_identity_sidecar_compact.npz`.
- HGNN WR uses focus-slot semantic MoE probabilities when a checkpoint exposes
  slot deltas; older checkpoints fall back to raw `final_logit` probabilities.
- Semantic group feature schema: v2, 25 compact per-slot features.
- Model-alignment rows score each slot with its focus-side win probability; blue
  slots use the blue-team frame and red slots use the mirrored red-team frame.
- Continuous thresholds are global side-row team-average percentiles.
- Count thresholds use explicit enemy-team counts.
- WR, effects, and gaps are focus-side win-rate percentage points.
- Accuracy is focus-row classification accuracy at the `0.5` threshold.
- `Acc if calibrated` shifts each bin's predictions so the bin mean equals the
  empirical WR while preserving within-bin ranking; this is diagnostic only.

## Gap Summary

The previous full report mixed high-support examples with sparse tail examples.
Those sparse examples inflated headline `Max Abs Gap`, including bins with
`n < 200`. This fixture now reports high-support examples only and delegates
aggregate calibration to the group EB audit. Regenerate the full split summary
from the current production checkpoint when refreshing this report; do not carry
forward historical max-gap or split-count tables after a cache refresh.

## Retained High-Support Examples

All retained examples had every populated bin at `n >= 500` on the previous
rendered validation audit. Refresh these counts before promotion review on the
rolled split.

| Example | Axis | Min bin n | Why keep it |
|---|---|---:|---|
| Yasuo TOP `crit` vs enemy siege | enemy siege pressure | 650 | Melee carry into poke/siege pressure with enough support across all bins. |
| Yasuo MIDDLE `crit` vs enemy siege | enemy siege pressure | 1,674 | Same pattern in a different role with much larger bins. |
| Ahri MIDDLE `ability_power` vs enemy scaling | enemy scaling pressure | 2,784 | Stable AP-mid scaling-context trajectory. |
| Malphite TOP `ar_tank` vs enemy physical | enemy physical share | 669 | Classic tank-into-physical interaction with direct semantic interpretability. |
| Sylas MIDDLE `ability_power` vs enemy range | enemy ranged count | 529 | Short-range battlemage range-pressure example above the support floor. |
| Kaisa BOTTOM any build vs enemy range | enemy ranged count | 1,793 | Bot-carry range-context example with strong support. |
| Kaisa BOTTOM `on_hit` vs enemy frontline count | enemy frontline count | 733 | Carry/build-specific frontline interaction. |
| Ahri MIDDLE `ability_power` vs enemy frontline count | enemy frontline count | 589 | AP mid into frontline context with enough per-bin volume. |
| Karma UTILITY any build vs enemy frontline count | enemy frontline count | 679 | Support-context example retained despite odd effect direction because bins are stable. |
| Viego JUNGLE any build vs enemy high-HP count | enemy high-HP count | 1,290 | High-HP enemy composition trajectory with broad support. |
| Malphite all roles `ar_tank` vs enemy physical | enemy physical share | 728 | Role-pooled armor-tank context. |
| Nautilus all roles `ar_tank` vs enemy physical | enemy physical share | 915 | Role-pooled armor-tank support example. |
| Darius TOP any build vs enemy range count | enemy ranged count | 662 | Juggernaut range-pressure context. |
| Darius TOP any build vs same-role range | lane-opponent range | 1,350 | Same-role range example with only stable bins retained. |
| Low own-damage teams vs enemy heal/shield | enemy heal/shield pressure | 10,473 | Team-level composition interaction with very large bins. |
| Ambessa TOP `attack_damage` vs enemy damage | enemy damage pressure | 1,140 | Fighter into damage-pressure trajectory. |
| Focus HP `<= 2309` vs enemy burst count | enemy burst count | 9,477 | Static durability into burst pressure with very large bins. |
| Focus HP `>= 2478` vs enemy burst count | enemy burst count | 10,375 | High-HP companion to the low-HP burst fixture. |
| Yasuo MIDDLE `crit` with ally CC | ally CC pressure | 1,135 | Ally setup context with stable support. |
| Jhin BOTTOM `crit` with ally CC | ally CC pressure | 889 | Marksman plus ally-CC trajectory. |
| Ezreal BOTTOM `attack_damage` vs enemy hard CC | enemy hard-CC count | 1,250 | Mobile carry into crowd-control pressure. |
| LeeSin JUNGLE `attack_damage` vs enemy scaling | enemy scaling pressure | 2,808 | Early-tempo jungler into scaling context. |
| Caitlyn BOTTOM `crit` vs enemy burst count | enemy burst count | 517 | Siege ADC into burst/dive pressure just above the support floor. |

## Removed Low-Support Examples

The following examples were removed from the detailed fixture set because at
least one populated bin had `n < 500`, making gap and max-gap readings too noisy
for headline interpretation:

- Graves JUNGLE `lethality` vs enemy damage.
- Nautilus UTILITY `mr_tank` with ally damage.
- Galio MIDDLE `mr_tank` vs enemy magic.
- Nilah BOTTOM any build vs enemy range.
- Sylas JUNGLE `ability_power` vs enemy frontline count.
- Sylas MIDDLE `ability_power` vs enemy frontline count.
- Vayne BOTTOM `on_hit` vs enemy frontline count.
- Thresh UTILITY `ar_tank` vs enemy burst count.
- Nautilus UTILITY `mr_tank` vs enemy burst count.
- Zed MIDDLE `lethality` vs enemy burst count.
- Nami UTILITY `utility_protection` vs enemy burst count.
- Jinx BOTTOM `crit` vs enemy burst count.
- Malphite TOP `ar_tank` vs heavy damage-taken count.
- Galio all roles `mr_tank` vs enemy magic.
- Nautilus all roles `mr_tank` vs enemy magic.
- MasterYi JUNGLE any build vs enemy hard CC.
- Selected enchanters UTILITY with skirmish allies.
- LeeSin JUNGLE `ad_off_tank` vs enemy magic.
- Thresh UTILITY `mr_tank` vs enemy magic.
- Ahri MIDDLE `ability_power` vs heavy damage-taken count.
- Kaisa BOTTOM `on_hit` vs heavy damage-taken count.
- Lulu UTILITY `utility_protection` with ally damage.
- Jayce TOP `attack_damage` vs enemy frontline count.

## Reproduction Commands

Regenerate predictions from the promoted production checkpoint when refreshing
the fixture:

```bash
uv run python -m app.ml.context_examples_audit \
  --context-cache-dir app/ml/data/cache \
  --model-cache-dir app/ml/data/cache \
  --model-path app/ml/data/hgnn_production_model.pt \
  --encoder-sidecar-path app/ml/data/semantic_identity_sidecar_compact.npz \
  --prediction-cache app/ml/data/audit_focus_side_probability.npy \
  --audit-split val \
  --output app/ml/documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md \
  --refresh-predictions
```

After regenerating, remove examples whose minimum populated bin count is below
`500` before using this document as a qualitative fixture.
