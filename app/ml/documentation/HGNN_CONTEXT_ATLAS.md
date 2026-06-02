# HGNN Context Atlas

Updated: 2026-06-02.

This doc describes the shared 24-dim context-atlas descriptor and head. It is
now the baseline/fallback path; production uses the threshold-tuned
identity-conditioned raw-atlas head documented in
[HGNN_IDENTITY_CONDITIONED_CONTEXT.md](HGNN_IDENTITY_CONDITIONED_CONTEXT.md).

The atlas replaces the retired hand-built 9-dim matchup profile with one
draft-safe descriptor for every `(championid, teamposition, build)` identity.
The model computes contexts from descriptors at inference time instead of
enumerating sparse matchup keys.

## Descriptor

`identity_context = [14 interpretable axes || 10 dense PCA axes]`.

The descriptor is keyed by identity and built from train-split aggregates only.
It never reads current-match post-game stats and never uses
`participant_challenges` or `challenge_*` features.

| Axis | Feature | Scale |
| ---: | --- | --- |
| 0 | `phys_offense_share` | `[0, 1]` share |
| 1 | `magic_offense_share` | `[0, 1]` share |
| 2 | `true_offense_share` | `[0, 1]` share |
| 3 | `armor_resist_frac` | `[0, 1]` fraction |
| 4 | `mr_resist_frac` | `[0, 1]` fraction |
| 5 | `champion_damage_pressure` | p95-scaled pressure |
| 6 | `phys_damage_pressure` | pressure x physical share |
| 7 | `magic_damage_pressure` | pressure x magic share |
| 8 | `true_damage_pressure` | pressure x true share |
| 9 | `damage_taken_pressure` | p95-scaled pressure |
| 10 | `heal_shield_pressure` | p95-scaled pressure |
| 11 | `cc_pressure` | p95-scaled pressure |
| 12 | `siege_pressure` | p95-scaled pressure |
| 13 | `scaling_pressure` | p95-scaled pressure |

Axes `0-8` are byte-for-byte the old matchup profile. Axes `9-13` cover context
families the profile could not express directly: durability, sustain, crowd
control, siege, and scaling. The 10-dim tail is PCA over allowed
`participant_stats`, timeline metrics, and derived ratios.

## Shared Context Head

The shared head scores every player against ally/enemy set summaries:

```text
feat_p = [self, enemy_mean, enemy_damage_weighted_mean, lane_opp,
          ally_mean, products(7)]
conf_p = support_p / (support_p + context_support_strength)
context_logit = sum_blue conf_p * head(feat_p) - sum_red conf_p * head(feat_p)
head = Linear(D -> 32) -> ReLU -> Linear(32 -> 1)
```

The seven products are:

| Product | Intent |
| --- | --- |
| `armor * enemy_phys` | armor value into physical teams |
| `mr * enemy_magic` | MR value into magic teams |
| `armor * (enemy_phys - enemy_magic)` | armor specialization |
| `mr * (enemy_magic - enemy_phys)` | MR specialization |
| `damage_taken * enemy_damage` | durability into high-damage teams |
| `own_damage * enemy_heal_shield` | damage into sustain |
| `own_heal_shield * ally_damage` | support/enchanter with carries |

The same head parameters score both teams. The final layer is zero-initialized,
support gates suppress rare identities, and `fwd - rev` is exactly
antisymmetric under `swap_hgnn_inputs`.

## Pipeline

```bash
python -m app.classification.embeddings.context
python -m app.ml.backfill_identity_context
python -m app.ml.train --shared-context
python -m app.ml.context_probes
pytest tests/ml/test_context_head.py tests/classification/test_identity_context.py
```

`build_dataset.py` writes `identity_context` and `identity_context_support` in a
full cache rebuild. `backfill_identity_context.py` writes the same arrays from
an existing cache without ClickHouse.

## Shared-Head Results

Shared 24-dim atlas, `1.15M` train games:

| Split | Acc | Thr Acc | AUC | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.5717 | 0.5696 | 0.6022 | 0.6751 | 0.2412 | 0.0063 |
| val | 0.5733 | 0.5780 | 0.6005 | 0.6754 | 0.2413 | 0.0278 |
| test | 0.5703 | 0.5735 | 0.5953 | 0.6770 | 0.2421 | 0.0244 |

Against the retired profile-head baseline, test AUC moved `0.5942 -> 0.5953`,
NLL `0.6773 -> 0.6770`, and Brier `0.2422 -> 0.2421`. The gain was modest but
the atlas generalized the context feature surface beyond one hand-tuned profile.

## Shared-Head Limitation

The shared atlas head was globally close to the 24-dim descriptor ceiling but
compressed identity-specific context sensitivities. Malphite `ar_tank` into
enemy physical share was the clearest example. Continuous bins use the same
global side-row quintile bands as
[HGNN_CONTEXT_EXAMPLES_AUDIT.md](HGNN_CONTEXT_EXAMPLES_AUDIT.md). Reproducer:
[../context_examples_audit.py](../context_examples_audit.py).

| Scope | Band | n | Emp WR | Shared model WR | Base WR | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all roles | `0-20` | 9,871 | 45.34 | 49.44 | 50.37 | +4.10 |
| all roles | `20-40` | 13,478 | 47.34 | 48.09 | 50.28 | +0.76 |
| all roles | `40-60` | 16,369 | 49.16 | 48.97 | 51.24 | -0.19 |
| all roles | `60-80` | 19,442 | 50.31 | 49.39 | 51.28 | -0.92 |
| all roles | `80-100` | 22,884 | 54.38 | 50.96 | 50.95 | -3.42 |
| TOP only | `0-20` | 7,161 | 44.88 | 49.53 | 50.49 | +4.64 |
| TOP only | `20-40` | 9,695 | 47.34 | 48.25 | 50.42 | +0.91 |
| TOP only | `40-60` | 12,449 | 49.47 | 49.12 | 51.40 | -0.35 |
| TOP only | `60-80` | 15,076 | 50.76 | 49.57 | 51.47 | -1.20 |
| TOP only | `80-100` | 17,269 | 54.35 | 50.97 | 51.01 | -3.37 |

The shared context head moved in the right direction but left a large
identity-specific tail gap. The identity-conditioned head narrows the all-role
`80-100` minus `0-20` delta gap from `-7.52` to `-4.44`.

Diagnosis: resistance fractions have low dynamic range across tank identities.
A single shared function cannot make Malphite much more sensitive to physical
enemies without also moving other tanks. The fix was to condition context
sensitivity on identity, not to add more global descriptor axes.

## Probe Results

Synthetic probes check direction, not empirical amplitude.

| Probe | Expected | Delta logit | Delta win-prob | Verdict |
| --- | --- | ---: | ---: | --- |
| Malphite `ar_tank` vs enemy physical | up | +0.0287 | +0.0072 | pass, under-scaled |
| DrMundo `mr_tank` vs enemy magic | up | +0.0291 | +0.0073 | pass |
| Galio `mr_tank` vs enemy magic | up | +0.0327 | +0.0082 | pass |
| Enchanter with carry/skirmish allies | up | +0.1568 | +0.0392 | pass |
| Low-damage identity vs enemy heal/shield | down | +0.0153 | +0.0038 | fail |
| Sion TOP `ad_off_tank` vs enemy damage | down | -0.0002 | -0.0001 | weak |

Amplitude is audited with real rows in
[HGNN_CONTEXT_EXAMPLES_AUDIT.md](HGNN_CONTEXT_EXAMPLES_AUDIT.md).

## Risks

| Risk | Control |
| --- | --- |
| Leakage | Identity-keyed train-split aggregates only; challenge and current-match post-game fields are forbidden. |
| Sparse identities | Per-player support gate, low-rank tail, and zero-init suppress rare noisy identities. |
| Overfit | Small zero-init residual, weight decay, and validation checkpointing. |
| Causality | Context responses are learned historical associations, not interventions. |
