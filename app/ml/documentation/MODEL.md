# ML Model Structure

Transformer over 10 fixed player tokens. Produces an antisymmetric blue-vs-red logit, optionally plus an antisymmetric prediction-band MoE residual. Current hyperparameters in [README.md](README.md); architecture sweep evidence in [OPTIMISATIONS.md](OPTIMISATIONS.md).

## Flow

```text
1. inputs:    champion_idx, role_idx, build_idx     shape [B, 10] each
2. embed:     tokens = champ_emb + role_emb + build_emb + side_emb   -> [B, 10, d]
              side is fixed: positions 0..4 = BLUE, positions 5..9 = RED
3. input:     tokens = Dropout(LayerNorm(tokens))
4. encoder:   for _ in range(n_layers):
                tokens = tokens + SelfAttention(LayerNorm(tokens))
                tokens = tokens + FFN(LayerNorm(tokens))            # Linear -> GELU -> Dropout -> Linear
              -> [B, 10, d]
5. split:     blue_tokens = tokens[:, 0:5]
              red_tokens  = tokens[:, 5:10]
6. pool:      b = pool(blue_tokens)                                  # [B, d]
              r = pool(red_tokens)                                   # [B, d]
              pool in { team_mean, team_attention }
7. match:     m(x, y) = concat(x, y, x - y, abs(x - y), x * y)       # [B, 5d]
8. head:      head(v) = Linear(GELU(Linear(LayerNorm(v))))           # 5d -> head_hidden -> 1
9. baseline:  baseline_logit = head(m(b, r)) - head(m(r, b))         # antisymmetric by construction
10. moe:      if use_moe:
                p = sigmoid(baseline_logit)
                route by triangular kernels over 40-60% baseline probability bands
                correction = moe(m(b, r), baseline_logit)
                           - moe(m(r, b), -baseline_logit)
              else:
                correction = 0
11. logit:    final_logit = baseline_logit + correction
12. output:   P(blue_win) = sigmoid(final_logit)
```

## Pooling Modes

| mode | per-team aggregation |
| --- | --- |
| `team_mean` | unweighted mean over the team's 5 tokens |
| `team_attention` | softmax-weighted sum, weights from `LayerNorm -> Linear(d -> 1)` per team |

## Notes

- Role index is implied by slot, not stored ([DATASET.md](DATASET.md)).
- The head is shared across both calls; antisymmetrisation is structural, not learned.
- MoE routing is anchored to baseline probability only. Experts see match features, but routing does not use champion, role, build, patch, temporal identity, or side.
- Expert bands are seven overlapping 5% windows centred at `0.425`, `0.450`, `0.475`, `0.500`, `0.525`, `0.550`, and `0.575`; samples outside 40-60% receive zero MoE correction.
- Expert residuals are bounded by `correction_scale * tanh(raw_delta)`, then antisymmetrized with the reverse match call.
