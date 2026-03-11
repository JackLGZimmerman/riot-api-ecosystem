# metrics

This directory defines metric specifications and an ABAC-style registry for applying metrics to targets.

## Registry Scope

Registry files:
- `registry/metrics.md`
- `registry/groups.md`
- `registry/individuals.md` 

Assignment templates:
- `scripts/assign_metric_to_group.sql`
- `scripts/assign_metric_to_individual.sql`
- `scripts/resolve_effective_metrics.sql`

## ABAC-Style Assignment Flow

Attributes used for resolution:
- `subject_type`: `individual` or `group`
- `subject_id`: target group/individual id
- `champion_id`: nullable
- `teamposition`: nullable
- `effect`: `allow` or `deny`
- `priority`: higher wins within same scope level

Resolution precedence:
1. individual + champion + teamposition
2. individual + champion
3. group + champion + teamposition
4. group + champion
5. global default

## Practical Example

Goal: apply metric `COMPOSITE_KDA_EARLY` to Ahri mid only, while denying it globally.

Inputs:
- Metric: `COMPOSITE_KDA_EARLY`
- Champion: Ahri (`champion_id=103`)
- Position: `MIDDLE`
- Individual: `champion:103`

Rules:
1. Global rule: `deny` `COMPOSITE_KDA_EARLY`
2. Individual champion rule: `allow` for `champion_id=103`, `teamposition=NULL`
3. Individual champion+position rule: `allow` for `champion_id=103`, `teamposition='MIDDLE'`

Outcomes:
- Ahri mid: allowed (rule 3 wins)
- Ahri top: allowed (rule 2 wins)
- Any other champion: denied (falls back to global rule 1)

## Runtime Flow

1. Build candidate rules for a metric and target context.
2. Match by attributes (`champion_id`, `teamposition`, `subject_type`, `subject_id`).
3. Sort by precedence and `priority`.
4. Take first row as effective rule.
5. Include/exclude metric based on final `effect`.

## Visualization

```text
Target Context
(champion_id=103, teamposition=MIDDLE)
        |
        v
Load candidate rules for metric
        |
        v
Rank candidates by precedence
  1) individual + champion + position
  2) individual + champion
  3) group + champion + position
  4) group + champion
  5) global
        |
        v
Tie-break: highest priority
        |
        v
Effective effect (allow/deny)
        |
        v
Metric included or excluded
```
