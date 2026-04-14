# timeline events + p_stats

This catalog contains combined timeline metrics derived from `events` and `p_stats`.

`p_stats` is an abbreviation for `participant_stats` in this metrics catalog. The underlying source table remains `game_data.tl_participant_stats`.

This file is intentionally same-frame only and does not use `t - (t-1)` style delta formulas.

Unless noted otherwise:
- `minute_bin = intDiv(frame_timestamp, 60000)`
- `events` minute bins are aligned to the `p_stats` frame snapshot in the same `minute_bin`
- `totalFarm = minionskilled + jungleminionskilled`
- `spentGold = totalgold - currentgold`
- `nonChampionDamageDone = totaldamagedone - totaldamagedonetochampions`
- Where kill-side event bins are reused, `killerid` is aligned to `participantid`, following the convention already documented in `events.md`.

| id | name | data_source | description | calculation | version | fields |
|---|---|---|---|---|---|---|
| TLX_S_001 | synthetic.tl_events_p_stats.championDamagePerDeathEvent | derived:TLE_S_025 + game_data.tl_participant_stats (see notes above) | Champion damage snapshot contextualized by death count in the aligned minute bin. | totaldamagedonetochampions / greatest(TLE_S_025, 1) grouped by (matchid, participantid, minute_bin) | 1.0.0 | totaldamagedonetochampions, TLE_S_025, frame_timestamp, participantid, matchid |

## Metric List

These are combined timeline metrics that join `events` bins with `p_stats` frame snapshots. All metrics in this file are derived and new.

### Derived Death-Normalized State

- [TLX_S_001] `championDamagePerDeathEvent` (new): `totaldamagedonetochampions / greatest(TLE_S_025, 1) grouped by (matchid, participantid, minute_bin)`
