# HGNN Identity-Conditioned Context

Updated: 2026-06-02.

The identity-conditioned context head lets each `(championid, teamposition,
build)` identity express its own context sensitivities without per-champion
rules or sparse matchup keys. It replaces the shared context-atlas residual when
enabled, and is now the production semantic-context path.

Production checkpoint: `app/ml/data/structured_winrate_model.pt`, selected by
validation threshold accuracy. AUC-selected reference:
`app/ml/data/experiments/identity_conditioned/cond_raw.pt`.

## Design

Principle: wide raw atlas, narrow regularized interaction.

| Component | Shape / value | Purpose |
| --- | --- | --- |
| `identity_context_raw` | 62 dims | 14 interpretable axes + 48 standardized draft-safe metrics |
| `identity_context` | 24 dims | shared descriptor; dense tail optional for `raw_plus_dense` |
| low-rank rank | 16 in best run | bottleneck for identity-specific context sensitivity |
| hidden dim | 64 in best run | conditioner/projector width |

Per player:

```text
context_feat_p  = [self, enemy_mean, enemy_weighted, lane_opp, ally_mean]
identity_cond_p = [champion_emb, role_emb, build_emb, self_raw]

z_id_p        = identity_conditioner(identity_cond_p)
z_ctx_p       = context_projector(context_feat_p)
raw_context_p = init_scale * dot(z_id_p, z_ctx_p)
conf_p        = support_p / (support_p + context_support_strength)

context_logit = sum_blue conf_p * raw_context_p
              - sum_red  conf_p * raw_context_p
```

The same module scores both teams, so the residual flips sign exactly under
team swap. The context projector is zero-initialized, so training starts from
the no-context baseline. Missing identities with zero raw context and zero
support contribute zero.

Enabled only when:

```text
use_identity_conditioned_context = true
identity_context_conditioning_type = "low_rank"
identity_context_raw_dim > 0
```

Otherwise the model falls back to the shared context head.

## Pipeline

```bash
python -m app.classification.embeddings.context
python -m app.ml.backfill_identity_context
python -m app.ml.train
python -m app.ml.context_atlas_audit
pytest tests/ml/test_identity_conditioned_context.py tests/classification/test_identity_context.py
```

Older caches without `identity_context_raw` still load, but the conditioned head
is unavailable until the raw array is backfilled.

## Global Results

Shared head before vs threshold-tuned identity-conditioned raw head after
(`1.15M` train games):

| Split | Variant | Acc | Thr Acc | AUC | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | shared | 0.5715 | 0.5677 | 0.6015 | 0.6754 | 0.2414 | 0.0043 |
| train | conditioned | 0.5735 | 0.5710 | 0.6038 | 0.6747 | 0.2410 | 0.0037 |
| val | shared | 0.5732 | 0.5771 | 0.6000 | 0.6755 | 0.2414 | 0.0266 |
| val | conditioned | 0.5750 | 0.5779 | 0.6014 | 0.6749 | 0.2411 | 0.0252 |
| test | shared | 0.5691 | 0.5720 | 0.5944 | 0.6773 | 0.2422 | 0.0235 |
| test | conditioned | 0.5717 | 0.5743 | 0.5972 | 0.6763 | 0.2418 | 0.0222 |

Test deltas: AUC `+0.0027`, NLL `-0.0010`, Brier `-0.0004`, threshold accuracy
`+0.0023`. Train-test AUC gap changes from `0.0070` to `0.0066`.

The AUC-selected raw reference reached test AUC `0.5979` and threshold accuracy
`0.5751`; production instead selects the checkpoint by threshold accuracy and
lands at test AUC `0.5972`, threshold accuracy `0.5743`. `raw_plus_dense`
reached test AUC `0.5977`, below raw-only, and overfit slightly more. Keep
`raw` as the primary source.

## Support Buckets

Sparse identities are not harmed. Buckets use combined sparse `1v1`/`2vX`
support on the test split.

| Bucket | n | AUC shared | AUC conditioned | NLL shared | NLL conditioned |
| --- | ---: | ---: | ---: | ---: | ---: |
| low_1_4 | 68,703 | 0.5994 | 0.6046 | 0.6755 | 0.6738 |
| medium_5_49 | 54,842 | 0.5934 | 0.5951 | 0.6779 | 0.6772 |
| high_50_plus | 19,586 | 0.5801 | 0.5815 | 0.6820 | 0.6815 |

## Identity Atlas Audit

For every high-support identity (atlas support `>=200`, focus-side games
`>=4,000`), `context_atlas_audit.py` selects the enemy/ally axis with the largest
empirical sensitivity and compares empirical, shared, and conditioned gradients.
Per-identity AUC/NLL is the less biased signal because the gradient axis is
selected after inspection.

| Metric | Value |
| --- | ---: |
| qualified identities | 471 |
| identities with AUC improvement | 97.9% |
| mean per-identity AUC, shared | 0.5988 |
| mean per-identity AUC, conditioned | 0.6028 |
| mean AUC delta | +0.0041 |
| identities with NLL improvement | 97.2% |
| mean NLL delta | -0.0014 |
| top-60 under-fit gap closed | 70% |
| top-30 under-fit gap closed | 83% |

Malphite is not a special-case rule. `54/TOP/ar_tank` (support `48,845`) improves
AUC `0.6002 -> 0.6062`, NLL `0.6764 -> 0.6744`, and selected-axis gap
`0.0477 -> 0.0316`.

Sample largest before-gaps:

| Champion id | Role | Build | Emp gradient | Shared gradient | Cond gradient | Gap shared | Gap conditioned | AUC shared | AUC conditioned |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 104 | JUNGLE | `lethality` | -0.117 | -0.053 | -0.061 | 0.064 | 0.057 | 0.645 | 0.652 |
| 150 | TOP | `ar_tank` | +0.077 | +0.023 | +0.025 | 0.054 | 0.052 | 0.577 | 0.580 |
| 777 | MIDDLE | `on_hit` | +0.075 | +0.012 | +0.029 | 0.063 | 0.046 | 0.641 | 0.648 |
| 3 | MIDDLE | `mr_tank` | +0.081 | +0.025 | +0.043 | 0.056 | 0.038 | 0.612 | 0.618 |
| 31 | TOP | `mr_tank` | -0.080 | -0.032 | -0.043 | 0.048 | 0.037 | 0.603 | 0.610 |

## Acceptance Check

| Check | Result |
| --- | --- |
| Draft safety | train-split identity aggregates only; no challenge or post-game fields |
| Swap behavior | exact antisymmetry under `swap_hgnn_inputs` |
| Global metrics | AUC/NLL/Brier improve on test |
| Sparse support | every support bucket improves |
| Breadth | 97.9% of high-support identities improve AUC |
| Disable path | one flag; shared head remains fallback |

## Verdict

Identity-conditioned atlas interactions help. The raw atlas plus low-rank
identity conditioning improves global metrics, closes many under-fit identity
tails, preserves sparse buckets, and remains draft-safe.
