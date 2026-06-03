# HGNN Context Examples Audit

Updated: 2026-06-03.

This audit joins the empirical focus-side context examples to the trained semantic HGNN predictions for the same cached games. Each bin reports `n / empirical WR / HGNN WR / gap`, where gap is `HGNN WR - empirical WR`. Zero gap is the target.

## Scope And Threshold Definitions

- Context source: `app/ml/data/cache` side-row arrays, all splits combined.
- HGNN model: `app/ml/data/experiments/semantic_context_compact_run/model.pt`.
- HGNN cache: `app/ml/data/experiments/semantic_context_compact_cache`.
- HGNN WR uses raw `final_logit` probabilities; report-only temperature scaling is not applied.
- Side rows audited: 2,862,626.
- Model-alignment rows score blue slots with `P(blue wins)` and red slots with `1 - P(blue wins)`.
- Continuous thresholds are global side-row team-average percentiles.
- Count thresholds use explicit enemy-team counts.
- WR, effects, and gaps are focus-side win-rate percentage points.
- Selected-enchanter probe uses Sona, Karma, Lulu, and Zilean in `UTILITY` with `utility_enchanter` or `utility_protection`.
- Low own-damage probe is anchored once per team side, then compared against the enemy heal/shield context.

| Axis | Low threshold | High threshold | Notes |
|---|---|---|---|
| Physical share | `<= 0.387` | `>= 0.557` | Team-average identity-context physical share. |
| Magic share | `<= 0.373` | `>= 0.549` | Team-average identity-context magic share. |
| Damage pressure | `<= 0.739` | `>= 0.813` | Team-average champion damage pressure. |
| Damage-taken pressure | `<= 0.639` | `>= 0.721` | Team-average damage-taken pressure. |
| Heal/shield pressure | `<= 0.028` | `>= 0.202` | Team-average ally heal/shield pressure. |
| CC pressure | `<= 0.374` | `>= 0.539` | Team-average crowd-control pressure. |
| Siege pressure | `<= 0.441` | `>= 0.530` | Team-average siege and structure pressure. |
| Scaling pressure | `<= 0.829` | `>= 0.863` | Team-average scaling pressure. |
| Burst-proxy count | `0` | `>= 3` | Enemy slots with slot damage pressure `>= 0.952` and a non-tank build. |
| Hard-CC count | `0` | `>= 3` | Enemy slots with slot CC pressure `>= 0.696`. |
| Tank/frontline count | `0` | `>= 3` | Enemy builds in `ar_tank`, `mr_tank`, `ad_off_tank`, or `ap_off_tank`. |
| Heavy damage-taken count | `0` | `>= 3` | Enemy slots with slot damage-taken pressure `>= 0.822`. |
| High-HP count | `0` | `>= 3` | Enemy champions with static level-18 HP `>= 2478.5`. |
| Focus HP tier | `<= 2309.0` | `>= 2478.5` | Static champion level-18 HP. |
| Ranged count | `<= 1` | `>= 4` | Static `attackRange_flat > 250` as ranged. |
| Same-role range | `<= 250` | `> 250` | Static attack range for the lane opponent. |
| Skirmish-ally count | `0` | `>= 2` | Gwen, Jax, Irelia, Fiora, Udyr, and XinZhao on the focus team. |

## Gap Summary

| Section | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE |
|---|---:|---:|---:|---:|---:|
| Headline Trajectory Audit Tables | 9 | 43 | 1.28 pp | 5.49 pp | 3.01 pp^2 |
| Richer Composition Trajectory Tables | 13 | 52 | 1.41 pp | 6.90 pp | 4.08 pp^2 |
| Retained Prior And User-Requested Trajectory Tables | 12 | 53 | 1.01 pp | 5.51 pp | 2.17 pp^2 |
| Inspected Lower-Signal Trajectory Tables | 4 | 16 | 0.72 pp | 4.39 pp | 1.74 pp^2 |

## Headline Trajectory Audit Tables

| Audit | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Bin 5 | Empirical effect | HGNN effect | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Yone TOP `on_hit` vs enemy siege | `<= 0.441`<br/>n=1,184<br/>emp=41.72%<br/>HGNN=40.69%<br/>gap=-1.04 pp | `0.441-0.471`<br/>n=956<br/>emp=35.88%<br/>HGNN=35.48%<br/>gap=-0.40 pp | `0.471-0.499`<br/>n=949<br/>emp=32.56%<br/>HGNN=32.99%<br/>gap=+0.43 pp | `0.499-0.530`<br/>n=830<br/>emp=33.01%<br/>HGNN=31.04%<br/>gap=-1.97 pp | `>= 0.530`<br/>n=845<br/>emp=25.92%<br/>HGNN=28.84%<br/>gap=+2.92 pp | -15.81 pp | -11.85 pp | Melee carry into siege and poke. |
| Graves JUNGLE `lethality` vs enemy damage | `<= 0.739`<br/>n=4,165<br/>emp=40.67%<br/>HGNN=38.67%<br/>gap=-2.00 pp | `0.739-0.764`<br/>n=3,431<br/>emp=32.96%<br/>HGNN=33.07%<br/>gap=+0.11 pp | `0.764-0.785`<br/>n=2,953<br/>emp=29.90%<br/>HGNN=31.43%<br/>gap=+1.53 pp | `0.785-0.813`<br/>n=2,842<br/>emp=28.68%<br/>HGNN=30.29%<br/>gap=+1.62 pp | `>= 0.813`<br/>n=2,284<br/>emp=26.05%<br/>HGNN=29.09%<br/>gap=+3.04 pp | -14.62 pp | -9.59 pp | Burst jungler into high enemy damage. |
| Yone MIDDLE `on_hit` vs enemy siege | `<= 0.441`<br/>n=1,349<br/>emp=42.92%<br/>HGNN=41.04%<br/>gap=-1.88 pp | `0.441-0.471`<br/>n=1,094<br/>emp=35.01%<br/>HGNN=35.34%<br/>gap=+0.33 pp | `0.471-0.499`<br/>n=1,139<br/>emp=32.92%<br/>HGNN=32.94%<br/>gap=+0.02 pp | `0.499-0.530`<br/>n=989<br/>emp=29.22%<br/>HGNN=31.67%<br/>gap=+2.45 pp | `>= 0.530`<br/>n=968<br/>emp=28.82%<br/>HGNN=29.57%<br/>gap=+0.75 pp | -14.10 pp | -11.47 pp | Same melee-carry pattern across lane. |
| Swain UTILITY `ap_off_tank` vs enemy scaling | `<= 0.829`<br/>n=1,151<br/>emp=50.48%<br/>HGNN=51.67%<br/>gap=+1.19 pp | `0.829-0.841`<br/>n=1,023<br/>emp=44.87%<br/>HGNN=46.45%<br/>gap=+1.58 pp | `0.841-0.852`<br/>n=988<br/>emp=44.74%<br/>HGNN=44.09%<br/>gap=-0.65 pp | `0.852-0.863`<br/>n=901<br/>emp=40.40%<br/>HGNN=42.90%<br/>gap=+2.50 pp | `>= 0.863`<br/>n=839<br/>emp=37.19%<br/>HGNN=42.68%<br/>gap=+5.49 pp | -13.29 pp | -8.99 pp | Drain support into scaling enemies. |
| Nautilus UTILITY `mr_tank` with ally damage | `<= 0.739`<br/>n=7,027<br/>emp=44.46%<br/>HGNN=45.18%<br/>gap=+0.72 pp | `0.739-0.764`<br/>n=7,616<br/>emp=47.65%<br/>HGNN=47.34%<br/>gap=-0.31 pp | `0.764-0.785`<br/>n=7,512<br/>emp=49.52%<br/>HGNN=49.05%<br/>gap=-0.47 pp | `0.785-0.813`<br/>n=7,393<br/>emp=50.90%<br/>HGNN=50.27%<br/>gap=-0.63 pp | `>= 0.813`<br/>n=2,888<br/>emp=55.16%<br/>HGNN=52.67%<br/>gap=-2.49 pp | +10.70 pp | +7.50 pp | Engage support with damage behind it. |
| Galio MIDDLE `mr_tank` vs enemy magic | `<= 0.373`<br/>n=2,045<br/>emp=38.63%<br/>HGNN=42.61%<br/>gap=+3.98 pp | `0.373-0.423`<br/>n=2,734<br/>emp=41.51%<br/>HGNN=43.14%<br/>gap=+1.63 pp | `0.423-0.486`<br/>n=3,300<br/>emp=41.52%<br/>HGNN=43.65%<br/>gap=+2.14 pp | `0.486-0.549`<br/>n=4,798<br/>emp=43.41%<br/>HGNN=43.41%<br/>gap=-0.00 pp | `>= 0.549`<br/>n=6,179<br/>emp=47.74%<br/>HGNN=47.58%<br/>gap=-0.16 pp | +9.11 pp | +4.97 pp | Anti-magic tank itemization. |
| Malphite TOP `ar_tank` vs enemy physical | `<= 0.387`<br/>n=7,668<br/>emp=44.98%<br/>HGNN=47.13%<br/>gap=+2.15 pp | `0.387-0.448`<br/>n=9,238<br/>emp=46.83%<br/>HGNN=47.48%<br/>gap=+0.65 pp | `0.448-0.508`<br/>n=12,809<br/>emp=49.44%<br/>HGNN=49.64%<br/>gap=+0.19 pp | `0.508-0.557`<br/>n=15,162<br/>emp=51.35%<br/>HGNN=51.37%<br/>gap=+0.02 pp | `>= 0.557`<br/>n=16,773<br/>emp=54.30%<br/>HGNN=53.98%<br/>gap=-0.31 pp | +9.32 pp | +6.85 pp | Armor tank into AD-heavy enemies. |
| Swain MIDDLE any build vs enemy range | `<= 1`<br/>n=1,802<br/>emp=57.77%<br/>HGNN=56.83%<br/>gap=-0.94 pp | `2`<br/>n=6,414<br/>emp=52.15%<br/>HGNN=53.34%<br/>gap=+1.19 pp | `3`<br/>n=7,056<br/>emp=50.35%<br/>HGNN=51.01%<br/>gap=+0.65 pp | `>= 4`<br/>n=1,717<br/>emp=48.51%<br/>HGNN=49.25%<br/>gap=+0.74 pp | N/A | -9.25 pp | -7.58 pp | Static range pressure on short-range battlemage. |
| Nilah BOTTOM any build vs enemy range | `<= 1`<br/>n=2,368<br/>emp=58.99%<br/>HGNN=56.81%<br/>gap=-2.18 pp | `2`<br/>n=9,098<br/>emp=54.21%<br/>HGNN=54.17%<br/>gap=-0.04 pp | `3`<br/>n=10,090<br/>emp=52.77%<br/>HGNN=52.68%<br/>gap=-0.09 pp | `>= 4`<br/>n=2,476<br/>emp=50.53%<br/>HGNN=52.16%<br/>gap=+1.63 pp | N/A | -8.47 pp | -4.66 pp | Melee bot lane into range-heavy teams. |

## Richer Composition Trajectory Tables

| Audit | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Bin 5 | Empirical effect | HGNN effect | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Swain BOTTOM `ability_power` vs enemy frontline count | `0`<br/>n=5,420<br/>emp=47.79%<br/>HGNN=49.33%<br/>gap=+1.55 pp | `1`<br/>n=9,537<br/>emp=50.98%<br/>HGNN=51.71%<br/>gap=+0.73 pp | `2`<br/>n=5,157<br/>emp=54.59%<br/>HGNN=54.94%<br/>gap=+0.35 pp | `>= 3`<br/>n=1,127<br/>emp=61.22%<br/>HGNN=59.71%<br/>gap=-1.52 pp | N/A | +13.44 pp | +10.38 pp | Swain gets better as enemies add durable targets. |
| Swain MIDDLE any build vs enemy frontline count | `0`<br/>n=4,318<br/>emp=47.99%<br/>HGNN=49.81%<br/>gap=+1.83 pp | `1`<br/>n=7,532<br/>emp=50.50%<br/>HGNN=51.54%<br/>gap=+1.03 pp | `2`<br/>n=4,221<br/>emp=55.86%<br/>HGNN=54.80%<br/>gap=-1.07 pp | `>= 3`<br/>n=918<br/>emp=58.61%<br/>HGNN=59.28%<br/>gap=+0.67 pp | N/A | +10.62 pp | +9.47 pp | Same Swain anti-frontline pattern mid. |
| Swain UTILITY any build vs enemy frontline count | `0`<br/>n=5,620<br/>emp=45.02%<br/>HGNN=46.14%<br/>gap=+1.13 pp | `1`<br/>n=9,781<br/>emp=46.98%<br/>HGNN=47.56%<br/>gap=+0.58 pp | `2`<br/>n=5,060<br/>emp=49.82%<br/>HGNN=49.98%<br/>gap=+0.16 pp | `>= 3`<br/>n=959<br/>emp=54.43%<br/>HGNN=54.01%<br/>gap=-0.42 pp | N/A | +9.41 pp | +7.86 pp | Support Swain also improves into frontline-heavy teams. |
| Lillia JUNGLE `ap_off_tank` vs enemy frontline count | `0`<br/>n=3,887<br/>emp=43.74%<br/>HGNN=45.04%<br/>gap=+1.30 pp | `1`<br/>n=7,222<br/>emp=46.36%<br/>HGNN=47.38%<br/>gap=+1.02 pp | `2`<br/>n=4,009<br/>emp=51.01%<br/>HGNN=50.42%<br/>gap=-0.59 pp | `>= 3`<br/>n=811<br/>emp=56.60%<br/>HGNN=54.93%<br/>gap=-1.66 pp | N/A | +12.86 pp | +9.90 pp | Sustained AP skirmisher into beefy teams. |
| Morgana UTILITY `ability_power` vs enemy frontline count | `0`<br/>n=6,554<br/>emp=45.65%<br/>HGNN=45.98%<br/>gap=+0.33 pp | `1`<br/>n=11,385<br/>emp=48.34%<br/>HGNN=47.74%<br/>gap=-0.60 pp | `2`<br/>n=5,581<br/>emp=51.98%<br/>HGNN=50.17%<br/>gap=-1.81 pp | `>= 3`<br/>n=1,081<br/>emp=56.71%<br/>HGNN=54.06%<br/>gap=-2.65 pp | N/A | +11.06 pp | +8.08 pp | Zone and control support benefits when enemies walk into space. |
| Vayne BOTTOM `on_hit` vs enemy frontline count | `0`<br/>n=6,867<br/>emp=48.45%<br/>HGNN=49.25%<br/>gap=+0.80 pp | `1`<br/>n=13,525<br/>emp=49.47%<br/>HGNN=50.56%<br/>gap=+1.09 pp | `2`<br/>n=7,822<br/>emp=52.26%<br/>HGNN=52.11%<br/>gap=-0.15 pp | `>= 3`<br/>n=1,685<br/>emp=59.29%<br/>HGNN=55.04%<br/>gap=-4.25 pp | N/A | +10.84 pp | +5.78 pp | Classic anti-tank marksman pattern. |
| Alistar UTILITY `ar_tank` vs enemy burst count | `0`<br/>n=7,064<br/>emp=53.67%<br/>HGNN=52.95%<br/>gap=-0.72 pp | `1`<br/>n=14,117<br/>emp=51.77%<br/>HGNN=51.69%<br/>gap=-0.08 pp | `2`<br/>n=5,754<br/>emp=50.16%<br/>HGNN=49.86%<br/>gap=-0.30 pp | `>= 3`<br/>n=791<br/>emp=40.96%<br/>HGNN=47.87%<br/>gap=+6.90 pp | N/A | -12.71 pp | -5.08 pp | Durable engage support punished by multiple burst threats. |
| Sion TOP `mr_tank` vs enemy burst count | `0`<br/>n=3,307<br/>emp=51.22%<br/>HGNN=51.44%<br/>gap=+0.22 pp | `1`<br/>n=6,146<br/>emp=49.77%<br/>HGNN=49.60%<br/>gap=-0.17 pp | `2`<br/>n=3,263<br/>emp=47.20%<br/>HGNN=47.36%<br/>gap=+0.17 pp | `>= 3`<br/>n=581<br/>emp=41.82%<br/>HGNN=45.36%<br/>gap=+3.53 pp | N/A | -9.40 pp | -6.08 pp | High-HP tank loses into concentrated burst threats. |
| Qiyana JUNGLE `lethality` vs enemy burst count | `0`<br/>n=10,534<br/>emp=49.17%<br/>HGNN=47.64%<br/>gap=-1.53 pp | `1`<br/>n=19,199<br/>emp=47.47%<br/>HGNN=46.27%<br/>gap=-1.20 pp | `2`<br/>n=7,731<br/>emp=45.62%<br/>HGNN=44.98%<br/>gap=-0.64 pp | `>= 3`<br/>n=1,021<br/>emp=39.37%<br/>HGNN=43.47%<br/>gap=+4.10 pp | N/A | -9.80 pp | -4.17 pp | Assassin jungler into enemy burst stacking. |
| Rell UTILITY `utility_protection` vs enemy burst count | `0`<br/>n=8,589<br/>emp=54.94%<br/>HGNN=53.69%<br/>gap=-1.26 pp | `1`<br/>n=15,585<br/>emp=51.69%<br/>HGNN=52.18%<br/>gap=+0.49 pp | `2`<br/>n=6,396<br/>emp=51.09%<br/>HGNN=50.86%<br/>gap=-0.24 pp | `>= 3`<br/>n=883<br/>emp=45.64%<br/>HGNN=48.96%<br/>gap=+3.32 pp | N/A | -9.30 pp | -4.73 pp | All-in support punished by burst-heavy enemies. |
| Corki BOTTOM `crit` vs enemy burst count | `0`<br/>n=10,342<br/>emp=54.27%<br/>HGNN=53.12%<br/>gap=-1.15 pp | `1`<br/>n=20,817<br/>emp=51.80%<br/>HGNN=51.70%<br/>gap=-0.11 pp | `2`<br/>n=9,139<br/>emp=49.94%<br/>HGNN=50.15%<br/>gap=+0.21 pp | `>= 3`<br/>n=1,285<br/>emp=45.37%<br/>HGNN=48.25%<br/>gap=+2.88 pp | N/A | -8.90 pp | -4.87 pp | Fragile carry into burst-heavy enemies. |
| Malphite TOP `ar_tank` vs heavy damage-taken count | `0`<br/>n=15,788<br/>emp=53.19%<br/>HGNN=52.87%<br/>gap=-0.32 pp | `1`<br/>n=29,695<br/>emp=50.09%<br/>HGNN=50.65%<br/>gap=+0.56 pp | `2`<br/>n=14,524<br/>emp=48.42%<br/>HGNN=48.57%<br/>gap=+0.15 pp | `>= 3`<br/>n=1,643<br/>emp=42.42%<br/>HGNN=46.16%<br/>gap=+3.74 pp | N/A | -10.76 pp | -6.71 pp | Armor tank loses into teams with multiple high-soak targets. |
| Poppy JUNGLE any build vs enemy high-HP count | `0`<br/>n=1,048<br/>emp=46.37%<br/>HGNN=48.59%<br/>gap=+2.22 pp | `1`<br/>n=2,200<br/>emp=48.55%<br/>HGNN=49.96%<br/>gap=+1.41 pp | `2`<br/>n=1,751<br/>emp=46.60%<br/>HGNN=51.10%<br/>gap=+4.50 pp | `>= 3`<br/>n=679<br/>emp=56.70%<br/>HGNN=52.62%<br/>gap=-4.09 pp | N/A | +10.33 pp | +4.03 pp | Anti-dash/control jungler into high-HP enemy teams. |

## Retained Prior And User-Requested Trajectory Tables

| Audit | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Bin 5 | Empirical effect | HGNN effect | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Malphite all roles `ar_tank` vs enemy physical | `<= 0.387`<br/>n=10,529<br/>emp=45.48%<br/>HGNN=47.48%<br/>gap=+2.00 pp | `0.387-0.448`<br/>n=13,017<br/>emp=46.88%<br/>HGNN=47.60%<br/>gap=+0.72 pp | `0.448-0.508`<br/>n=16,829<br/>emp=49.06%<br/>HGNN=49.78%<br/>gap=+0.71 pp | `0.508-0.557`<br/>n=19,526<br/>emp=50.86%<br/>HGNN=51.34%<br/>gap=+0.48 pp | `>= 0.557`<br/>n=22,143<br/>emp=54.42%<br/>HGNN=54.04%<br/>gap=-0.38 pp | +8.94 pp | +6.56 pp | Original armor-stack audit, retained beyond TOP-only. |
| Galio all roles `mr_tank` vs enemy magic | `<= 0.373`<br/>n=2,182<br/>emp=39.05%<br/>HGNN=42.97%<br/>gap=+3.92 pp | `0.373-0.423`<br/>n=2,969<br/>emp=41.93%<br/>HGNN=43.50%<br/>gap=+1.57 pp | `0.423-0.486`<br/>n=3,637<br/>emp=42.10%<br/>HGNN=44.18%<br/>gap=+2.09 pp | `0.486-0.549`<br/>n=5,566<br/>emp=43.96%<br/>HGNN=44.04%<br/>gap=+0.07 pp | `>= 0.549`<br/>n=7,642<br/>emp=48.59%<br/>HGNN=48.47%<br/>gap=-0.12 pp | +9.54 pp | +5.50 pp | Original anti-magic tank family, broader than MIDDLE-only. |
| Chogath all roles `mr_tank` vs enemy magic | `<= 0.373`<br/>n=486<br/>emp=46.30%<br/>HGNN=50.40%<br/>gap=+4.11 pp | `0.373-0.423`<br/>n=665<br/>emp=44.96%<br/>HGNN=50.47%<br/>gap=+5.51 pp | `0.423-0.486`<br/>n=1,133<br/>emp=49.78%<br/>HGNN=50.47%<br/>gap=+0.69 pp | `0.486-0.549`<br/>n=2,076<br/>emp=49.47%<br/>HGNN=50.45%<br/>gap=+0.98 pp | `>= 0.549`<br/>n=3,353<br/>emp=53.50%<br/>HGNN=53.54%<br/>gap=+0.04 pp | +7.21 pp | +3.14 pp | Smaller support, but unique scaling-tank anti-magic case. |
| Nautilus all roles `ar_tank` vs enemy physical | `<= 0.387`<br/>n=10,135<br/>emp=46.82%<br/>HGNN=47.53%<br/>gap=+0.71 pp | `0.387-0.448`<br/>n=14,076<br/>emp=46.57%<br/>HGNN=46.79%<br/>gap=+0.22 pp | `0.448-0.508`<br/>n=18,450<br/>emp=48.50%<br/>HGNN=48.57%<br/>gap=+0.06 pp | `0.508-0.557`<br/>n=19,707<br/>emp=49.60%<br/>HGNN=49.22%<br/>gap=-0.37 pp | `>= 0.557`<br/>n=21,520<br/>emp=51.56%<br/>HGNN=50.78%<br/>gap=-0.78 pp | +4.74 pp | +3.25 pp | Physical-heavy enemy teams remain a support-tank check. |
| Darius TOP any build vs enemy range count | `<= 1`<br/>n=7,151<br/>emp=52.38%<br/>HGNN=52.17%<br/>gap=-0.22 pp | `2`<br/>n=28,731<br/>emp=49.50%<br/>HGNN=50.37%<br/>gap=+0.88 pp | `3`<br/>n=30,943<br/>emp=49.41%<br/>HGNN=49.24%<br/>gap=-0.17 pp | `>= 4`<br/>n=7,690<br/>emp=47.71%<br/>HGNN=48.18%<br/>gap=+0.47 pp | N/A | -4.67 pp | -3.98 pp | Static team range pressure, stronger than lane-only range. |
| Darius TOP any build vs same-role range | `<= 250`<br/>n=63,492<br/>emp=49.98%<br/>HGNN=50.06%<br/>gap=+0.07 pp | `> 250`<br/>n=11,023<br/>emp=47.09%<br/>HGNN=48.67%<br/>gap=+1.57 pp | N/A | N/A | N/A | -2.89 pp | -1.39 pp | User-requested static melee/ranged lane audit. |
| MasterYi JUNGLE any build vs enemy hard CC | `0`<br/>n=17,755<br/>emp=53.24%<br/>HGNN=54.12%<br/>gap=+0.88 pp | `1`<br/>n=27,265<br/>emp=52.22%<br/>HGNN=52.87%<br/>gap=+0.65 pp | `2`<br/>n=12,667<br/>emp=51.29%<br/>HGNN=51.84%<br/>gap=+0.55 pp | `>= 3`<br/>n=2,544<br/>emp=50.63%<br/>HGNN=51.86%<br/>gap=+1.23 pp | N/A | -2.61 pp | -2.26 pp | User-requested low-CC audit; unique even though gap is modest. |
| Selected enchanters UTILITY with skirmish allies | `0`<br/>n=382,693<br/>emp=50.30%<br/>HGNN=50.68%<br/>gap=+0.37 pp | `1`<br/>n=73,328<br/>emp=52.24%<br/>HGNN=52.06%<br/>gap=-0.18 pp | `>= 2`<br/>n=3,338<br/>emp=52.97%<br/>HGNN=53.00%<br/>gap=+0.03 pp | N/A | N/A | +2.66 pp | +2.32 pp | Original enchanter-with-skirmishers synergy probe. |
| Low own-damage teams vs enemy heal/shield | `<= 0.028`<br/>n=114,704<br/>emp=49.67%<br/>HGNN=50.44%<br/>gap=+0.76 pp | `0.028-0.077`<br/>n=116,203<br/>emp=48.25%<br/>HGNN=48.88%<br/>gap=+0.63 pp | `0.077-0.200`<br/>n=111,021<br/>emp=47.38%<br/>HGNN=47.54%<br/>gap=+0.16 pp | `0.200-0.202`<br/>n=120,704<br/>emp=47.52%<br/>HGNN=47.55%<br/>gap=+0.03 pp | `>= 0.202`<br/>n=117,404<br/>emp=47.48%<br/>HGNN=47.72%<br/>gap=+0.23 pp | -2.19 pp | -2.72 pp | Original low-damage into sustain audit. |
| Sion TOP `ad_off_tank` vs enemy damage | `<= 0.739`<br/>n=956<br/>emp=55.33%<br/>HGNN=53.09%<br/>gap=-2.24 pp | `0.739-0.764`<br/>n=938<br/>emp=53.94%<br/>HGNN=51.83%<br/>gap=-2.11 pp | `0.764-0.785`<br/>n=912<br/>emp=51.86%<br/>HGNN=51.24%<br/>gap=-0.62 pp | `0.785-0.813`<br/>n=1,090<br/>emp=51.19%<br/>HGNN=51.91%<br/>gap=+0.72 pp | `>= 0.813`<br/>n=1,022<br/>emp=53.91%<br/>HGNN=52.39%<br/>gap=-1.53 pp | -1.42 pp | -0.71 pp | Retained as a tank-into-damage pressure sanity check. |
| DrMundo all roles `ad_off_tank` vs enemy magic | `<= 0.373`<br/>n=1,390<br/>emp=62.37%<br/>HGNN=60.62%<br/>gap=-1.75 pp | `0.373-0.423`<br/>n=1,467<br/>emp=60.67%<br/>HGNN=60.68%<br/>gap=+0.01 pp | `0.423-0.486`<br/>n=1,593<br/>emp=62.59%<br/>HGNN=60.58%<br/>gap=-2.01 pp | `0.486-0.549`<br/>n=1,855<br/>emp=58.33%<br/>HGNN=59.42%<br/>gap=+1.09 pp | `>= 0.549`<br/>n=1,701<br/>emp=63.67%<br/>HGNN=61.50%<br/>gap=-2.17 pp | +1.29 pp | +0.88 pp | Original Mundo magic-share probe, low gap but distinct champion. |
| DrMundo all roles `mr_tank` vs enemy magic | `<= 0.373`<br/>n=1,540<br/>emp=51.23%<br/>HGNN=50.16%<br/>gap=-1.07 pp | `0.373-0.423`<br/>n=2,134<br/>emp=50.89%<br/>HGNN=50.13%<br/>gap=-0.76 pp | `0.423-0.486`<br/>n=3,101<br/>emp=49.08%<br/>HGNN=49.76%<br/>gap=+0.68 pp | `0.486-0.549`<br/>n=5,532<br/>emp=48.70%<br/>HGNN=49.75%<br/>gap=+1.05 pp | `>= 0.549`<br/>n=8,072<br/>emp=52.12%<br/>HGNN=52.99%<br/>gap=+0.87 pp | +0.88 pp | +2.83 pp | Retained to compare MR-tank Mundo against Galio/Chogath. |

## Inspected Lower-Signal Trajectory Tables

| Audit | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Bin 5 | Empirical effect | HGNN effect | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Focus HP `<= 2309` vs enemy burst count | `0`<br/>n=880,508<br/>emp=51.83%<br/>HGNN=51.82%<br/>gap=-0.01 pp | `1`<br/>n=1,544,006<br/>emp=50.68%<br/>HGNN=50.81%<br/>gap=+0.12 pp | `2`<br/>n=634,311<br/>emp=49.54%<br/>HGNN=49.59%<br/>gap=+0.05 pp | `>= 3`<br/>n=85,831<br/>emp=47.62%<br/>HGNN=47.81%<br/>gap=+0.19 pp | N/A | -4.21 pp | -4.01 pp | Broad HP-vs-burst check; useful but lower signal than champion-specific rows. |
| Focus HP `>= 2478` vs enemy burst count | `0`<br/>n=1,042,274<br/>emp=50.70%<br/>HGNN=50.49%<br/>gap=-0.21 pp | `1`<br/>n=1,844,824<br/>emp=49.44%<br/>HGNN=49.31%<br/>gap=-0.13 pp | `2`<br/>n=755,241<br/>emp=48.04%<br/>HGNN=48.13%<br/>gap=+0.09 pp | `>= 3`<br/>n=102,599<br/>emp=46.08%<br/>HGNN=46.43%<br/>gap=+0.35 pp | N/A | -4.62 pp | -4.06 pp | High-HP slots also drop into burst stacks, so champion/build specificity matters. |
| Swain MIDDLE any build vs heavy damage-taken count | `0`<br/>n=3,997<br/>emp=51.31%<br/>HGNN=51.67%<br/>gap=+0.36 pp | `1`<br/>n=8,307<br/>emp=51.13%<br/>HGNN=51.97%<br/>gap=+0.84 pp | `2`<br/>n=4,192<br/>emp=52.91%<br/>HGNN=53.20%<br/>gap=+0.29 pp | `>= 3`<br/>n=493<br/>emp=51.93%<br/>HGNN=56.31%<br/>gap=+4.39 pp | N/A | +0.61 pp | +4.65 pp | Swain into heavy damage-taken count was inspected; tank/frontline count is much stronger. |
| Swain BOTTOM `ability_power` vs heavy damage-taken count | `0`<br/>n=4,830<br/>emp=52.09%<br/>HGNN=51.92%<br/>gap=-0.17 pp | `1`<br/>n=10,201<br/>emp=51.17%<br/>HGNN=52.06%<br/>gap=+0.89 pp | `2`<br/>n=5,554<br/>emp=51.82%<br/>HGNN=52.84%<br/>gap=+1.03 pp | `>= 3`<br/>n=656<br/>emp=52.29%<br/>HGNN=54.64%<br/>gap=+2.35 pp | N/A | +0.20 pp | +2.72 pp | Same result bot: tank/frontline count is the better Swain audit. |

## Overall Summary

| Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE |
|---:|---:|---:|---:|---:|
| 38 | 164 | 1.18 pp | 6.90 pp | 2.96 pp^2 |

Gap MSE is `mean((HGNN_focus_WR - empirical_focus_WR)^2)` across populated threshold bins, rendered as percentage-points squared.
