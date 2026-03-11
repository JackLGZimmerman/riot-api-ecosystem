# individuals

Purpose: Direct individual targets for metric assignment.

Suggested columns:

| individual_id | individual_kind | external_id | champion_id | teamposition | name | status |
|---|---|---|---|---|---|---|

Notes:
- `champion_id` can be set for champion-specific assignment.
- `teamposition` can be `NULL` (all positions) or a value like `TOP`, `JUNGLE`, `MIDDLE`, `BOTTOM`, `UTILITY`.
- To target a champion in a specific role, set both `champion_id` and `teamposition`.
