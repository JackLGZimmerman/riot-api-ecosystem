# groups

Purpose: Reusable metric-target groups (generic, not champion-specific).

Suggested columns:

| group_id | group_kind | name | champion_id | teamposition | description | status |
|---|---|---|---|---|---|---|

Notes:
- Keep groups generic, but allow optional scope keys.
- `champion_id` + `teamposition` enables reusable scoped groups, for example "mid-lane assassins".
