# Classification Embeddings

Per-identity classification at the `(championid, teamposition, build)` key. The
pipeline runs a set of **specialist** embeddings — each asks one narrow
behavioural question over its own small feature subset and emits a label per
identity.

Run:

```bash
uv run python -m app.classification.embeddings.pipeline       # run specialists + report
uv run python -m app.classification.embeddings.specialists    # specialists only
uv run python -m app.classification.embeddings.tune           # specialist sweep
```

Source data is `game_data_filtered.synergy_1vx_temporal` (`6010`), smoothed
with the `9000-9040` prior tables. Embeddings are L2-normalised and grouped by
average-link agglomerative clustering on cosine distance.

## Specialists

Each specialist is a separate embedding whose feature set is chosen for the
independent directions retained by PCA. Groups whose median pairwise cosine
sits below `min_median_sim` or whose size is below `min_group_size=3` are
dropped (the identity gets a `-1` label).

Active registry — see `SPECIALISTS` in [embeddings/config.py](embeddings/config.py):

| Specialist | Question | Features |
| --- | --- | --- |
| `durability` | Damage absorbed, mitigated, healed, and incoming damage mix. | `all_sources_taken_to_death_ratio`, `all_sources_taken_to_gold_ratio`, `self_heal_to_taken_ratio`, `self_mitigation_to_total_taken_ratio`, `magic_damage_to_total_taken_ratio`, `physical_damage_to_total_taken_ratio` |
| `sustained_damage` | Sustained damage typing and efficiency. | `physical_damage_ratio`, `damage_to_taken_to_death_ratio`, `damage_dealt_to_gold_ratio` |
| `burst_damage` | Solo-killer vs team-fight contribution × crit reliance, with gold-efficiency on the K/A side. Five groups: crit AD carries, AP burst (gold-efficient), crit-utility (Senna/Ashe), AP zone mages, and tanks/enchanters. | `largestcriticalstrike`, `burst_kills_to_assists_ratio`, `burst_kills_to_assists_to_gold_ratio` |
| `vision` | Vision intensity × ward-sweep behaviour × vision-per-action efficiency. Five groups: supports (vision/gold +1.6), junglers + roamers (sweep +1.6, per-action +1.4, Urgot/Shaco/Qiyana/Talon), off-role enchanters in MID/TOP (low-intensity place-not-sweep), Fiddlesticks Scarecrow-Effigy anomaly (per-action +4.5), and a low-vision lane continuum (~54% — no positive read). | `visionscore_to_gold_ratio`, `visionscore_to_wards_placed_killed_ratio`, `wards_killed_to_placed_ratio` |
| `farming` | Lane-CS × jungle-CS volume crossed with CS-to-gold and CS-to-deaths efficiency. PCA splits balanced ~53/45 between volume and efficiency axes. Eight groups: lane farmers (~47% continuum — high CS, no further distinction beyond what the base build partition supplies), supports, junglers, off-role lane hybrids (Pantheon/Sylas/Soraka/Twitch/Lulu), lane-playing carry junglers (Shyvana/Udyr/Kindred/Belveth), roaming supports (Alistar/Sona/Janna/Bard), jungle champs picked as support (Viego/Belveth/Briar in UTILITY), and aggressive solo-lane carries (Qiyana/Naafiri/Khazix/Trundle). | `totalminionskilled`, `neutralminionskilled`, `total_farm_to_gold_ratio`, `total_farm_to_deaths_ratio` |

Previously registered specialists kept as reference in the config header for
re-introduction: `temporal`, `damage_profile`, `burst`, `team_utility`,
`crowd_control`, `engage`, `objective`, `economy`.

### Composition

Per-specialist labels are saved as `npz` files in
`data/embeddings/cache/specialists/<name>.npz` with `keys`, `key_columns`, and
`labels` arrays (`-1` for identities that fell into a dropped group).
Downstream code intersects labels across specialists.

### Adding A Specialist

1. Add a `SpecialistSpec` to `SPECIALISTS` in [embeddings/config.py](embeddings/config.py).
2. If the spec needs a derived feature not yet in `DERIVED_METRIC_FUNCS`,
   add it there.
3. Sweep `kv` × `t` with `tune.py --name <name>`.
4. Run `uv run python -m app.classification.embeddings.specialists` and
   inspect the report.

Prefer features that add a unique axis for the specialist. Raw metrics are
allowed, but avoid pairing a raw numerator with a ratio that already contains
the same information unless the PCA inspection shows a distinct retained
direction.

See [EXPERIMENTS.md](EXPERIMENTS.md) for tuning workflow.
