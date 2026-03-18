# LoL Dynamic Metric Model

## Goal

Use Python as the control plane and ClickHouse as the execution plane.

That means:

- Python decides what the selected group means.
- Python resolves which members and metrics are relevant.
- ClickHouse builds the final solution table from that resolved plan.

We do not want a permanent SQL control plane anymore.

## Architecture

### Python Control Plane

Python owns:

- group definitions
- group compositions
- member definitions
- group memberships
- metric definitions
- group metric assignments
- metric dependency rules
- composition validation rules
- selected-group resolution

Python should output one concrete execution plan for a selected group.

### ClickHouse Execution Plane

ClickHouse owns:

- source participant data
- source match and timeline data
- metric computation over large row sets
- joins against the resolved execution plan
- aggregation
- persistence of the final solution table

## Key Principle

The selected group should first become a resolved plan.

Only after the plan is fully resolved should ClickHouse build the final solution table.

So the runtime is:

1. Python resolves metadata.
2. Python produces a resolved plan.
3. ClickHouse executes that plan.
4. ClickHouse writes the final solution table.

## Logical Metadata Model

The structures below are logical control-plane entities.

They can live in:

- Python dictionaries
- Python classes
- YAML or JSON files loaded by Python
- another metadata store that Python reads

The examples are shown as small tables only to make the relationships clear.

### 1) Group Definitions

| group_id | group_name | group_type | meaning |
|---|---|---|---|
| `grp_support` | Support | `multi_member_group` | All support-role participants |
| `grp_support_engage` | Support Engage | `multi_member_group` | Engage-oriented supports |
| `grp_support_leona` | Leona Support | `champion_role` | Leona played as support |

### 2) Group Compositions

| parent_group_id | child_group_id | meaning |
|---|---|---|
| `grp_support` | `grp_support_engage` | Engage support is a child branch of support |
| `grp_support_engage` | `grp_support_leona` | Leona support is a child branch of engage support |

### 3) Members

| member_id | team_position | champion_id | build_scope | meaning |
|---|---|---|---|---|
| `mem_support_any` | `UTILITY` | `ANY` | `ANY` | Any support player |
| `mem_leona_support` | `UTILITY` | `89` | `ANY` | Leona support |
| `mem_nautilus_support` | `UTILITY` | `111` | `ANY` | Nautilus support |

### 4) Group Memberships

| group_id | member_id | why it is attached |
|---|---|---|
| `grp_support` | `mem_support_any` | broad support coverage |
| `grp_support_engage` | `mem_leona_support` | Leona is an engage support |
| `grp_support_engage` | `mem_nautilus_support` | Nautilus is an engage support |
| `grp_support_leona` | `mem_leona_support` | champion-specific branch |

### 5) Metric Definitions

| metric_id | metric_name | metric_kind | default_aggregation | meaning |
|---|---|---|---|---|
| `vision_score_per_min` | Vision Score Per Min | `existing` | `avg` | Broad support vision metric |
| `cc_score_per_min` | CC Score Per Min | `derived` | `avg` | Engage crowd control metric |
| `engage_success_rate` | Engage Success Rate | `composite` | `avg` | Engage outcome metric |
| `damage_taken_per_min` | Damage Taken Per Min | `existing` | `avg` | Tankiness metric |

### 6) Group Metric Assignments

| group_id | metric_id | why it is attached |
|---|---|---|
| `grp_support` | `vision_score_per_min` | every support should expose vision |
| `grp_support_engage` | `cc_score_per_min` | engage branch metric |
| `grp_support_engage` | `engage_success_rate` | engage branch metric |
| `grp_support_leona` | `damage_taken_per_min` | Leona tank profile |

### 7) Metric Dependencies

This is needed for higher-level metrics.

| metric_id | depends_on_metric_id | role |
|---|---|---|
| `engage_success_rate` | `engage_attempts` | denominator |
| `engage_success_rate` | `successful_engages` | numerator |

## Build Order

Build the control plane from the bottom up:

1. Define canonical metrics.
2. Define the most specific leaf groups.
3. Define the members for those leaf groups.
4. Attach metrics to those leaf groups.
5. Compose broader parent groups from the leaves.
6. Define composite metric dependencies.
7. Resolve one selected group into one execution plan.

## Member Resolution

Python should resolve members before any ClickHouse query is built.

### Segmentation Groups

These match directly against participant rows.

Example:

- support players
- Leona support

Recommended matching logic:

- exact role or `ANY`
- exact champion or `ANY`
- exact build or `ANY`

If multiple rules in the same branch match the same participant, keep one deterministic winner.

Recommended specificity order:

1. role + champion + build
2. role + champion
3. role only
4. any role / any champion

Preserve:

- `selected_group_id`
- `source_group_id`
- `matched_member_id`

### Composition Groups

These are not single-row matches.

Example:

- top Renekton plus any jungle
- engage support plus early-game jungle

Recommended composition logic:

1. Match the underlying member rules first.
2. Group those matches by match and team.
3. Check whether all required members exist together.
4. Emit a valid group instance only if the whole composition is present.

So composition groups are:

- first member matching
- then composition validation

## Metric Resolution

Python should resolve metrics by `metric_id`, not by direct source-column name at runtime.

Default rule:

- always add all metrics declared along the resolved branch path
- resolve from parent to leaf
- if the same `metric_id` is declared more than once, the definition closest to the leaf wins

So for one branch:

- parent metrics provide the generic layer
- child metrics add more specific meaning
- leaf metrics add the most specific meaning
- leaf-level conflicts override parent-level conflicts

Recommended formula:

- `effective_metrics(branch) = ordered union of declared metrics from root to leaf`
- `closest declaration to the leaf wins on duplicate metric_id`

Recommended metric categories:

- `existing`: already available in source data
- `derived`: computed directly from source fields
- `composite`: built from lower metrics after those metrics exist

Each metric definition should provide:

- `metric_id`
- `metric_name`
- `metric_kind`
- `source_grain`
- `default_aggregation`
- dependency list if composite

Python should resolve:

1. which metrics are allowed for each branch
2. which metrics must be built first
3. which metrics can be returned directly from source data

Metric names should come from the metric definition metadata, not from ad hoc SQL aliases.

### Sibling Groups At The Same Level

Sibling groups should be resolved independently by default.

Example:

- `grp_support_engage`
- `grp_support_enchanter`

Do not intersect sibling metrics as the base rule.

Default behavior:

- resolve each sibling branch separately
- let each sibling keep its own effective metric set
- preserve `source_group_id` so branch provenance is never lost

Optional behavior:

- if a reporting or modeling consumer needs one shared comparable metric set across siblings, derive a comparison view using intersection

Recommended formula for optional compare mode:

- `comparable_metrics(sibling_set) = intersection of effective_metrics for each sibling branch`

This compare-mode intersection is a derived output, not the base execution rule.

## Resolved Execution Plan

The output of the Python control plane should be one concrete resolved plan for the selected group.

Recommended plan components:

### Branch Plan

| selected_group_id | source_group_id | depth |
|---|---|---|
| `grp_support` | `grp_support` | `0` |
| `grp_support` | `grp_support_engage` | `1` |
| `grp_support` | `grp_support_leona` | `2` |

### Member Plan

| selected_group_id | source_group_id | matched_member_id | team_position | champion_id |
|---|---|---|---|---|
| `grp_support` | `grp_support` | `mem_support_any` | `UTILITY` | `ANY` |
| `grp_support` | `grp_support_engage` | `mem_leona_support` | `UTILITY` | `89` |
| `grp_support` | `grp_support_engage` | `mem_nautilus_support` | `UTILITY` | `111` |
| `grp_support` | `grp_support_leona` | `mem_leona_support` | `UTILITY` | `89` |

### Metric Plan

| selected_group_id | source_group_id | metric_id | metric_name | metric_kind |
|---|---|---|---|---|
| `grp_support` | `grp_support` | `vision_score_per_min` | Vision Score Per Min | `existing` |
| `grp_support` | `grp_support_engage` | `vision_score_per_min` | Vision Score Per Min | `existing` |
| `grp_support` | `grp_support_engage` | `cc_score_per_min` | CC Score Per Min | `derived` |
| `grp_support` | `grp_support_engage` | `engage_success_rate` | Engage Success Rate | `composite` |
| `grp_support` | `grp_support_leona` | `vision_score_per_min` | Vision Score Per Min | `existing` |
| `grp_support` | `grp_support_leona` | `cc_score_per_min` | CC Score Per Min | `derived` |
| `grp_support` | `grp_support_leona` | `engage_success_rate` | Engage Success Rate | `composite` |
| `grp_support` | `grp_support_leona` | `damage_taken_per_min` | Damage Taken Per Min | `existing` |

This resolved plan is what ClickHouse should execute against.

## Method To Build The Final Solution Table

This is the recommended method.

### Step 1: Resolve the Selected Group in Python

Input:

- `selected_group_id`

Python resolves:

- all descendant branches
- all effective member rules
- all effective metrics
- metric dependency order

### Step 2: Materialize the Resolved Plan for ClickHouse

Python passes the resolved plan into ClickHouse.

This can be done with:

- temporary tables
- staging tables
- external table inputs
- dataframe upload before query execution

This is execution input, not a permanent SQL control plane.

### Step 3: Read the Source Rows

ClickHouse reads the source data at a stable grain.

Recommended base grain:

- one participant in one match

### Step 4: Match Source Rows to the Resolved Member Plan

ClickHouse joins source rows to the resolved member rules and returns matched participants.

Recommended intermediate output:

| selected_group_id | source_group_id | matched_member_id | match_id | participant_id |
|---|---|---|---|---|
| `grp_support` | `grp_support` | `mem_support_any` | `1001` | `6` |
| `grp_support` | `grp_support_engage` | `mem_leona_support` | `1001` | `6` |
| `grp_support` | `grp_support_leona` | `mem_leona_support` | `1001` | `6` |

### Step 5: Build or Return Metrics

For each matched row:

- return `existing` metrics directly
- compute `derived` metrics from source fields
- compute `composite` metrics from dependency outputs

All metric outputs should share one standard shape.

### Step 6: Write the Final Solution Table

Recommended final solution table shape:

| selected_group_id | source_group_id | matched_member_id | group_instance_id | match_id | participant_id | metric_id | metric_name | metric_value |
|---|---|---|---|---|---|---|---|---|

Notes:

- `group_instance_id` can be null for simple segmentation groups.
- `group_instance_id` is useful for true composition groups.

This is the first correct target.

It is long-form, branch-aware, and stable.

### Step 7: Derive Optional Outputs

Once the final solution table is correct, you can derive:

- aggregated summary tables
- wide dashboard tables
- feature tables for downstream modeling

Those should be derived outputs, not the base representation.

## Example Setup

Selected group:

- `grp_support`

### Example Source Participant Rows

| match_id | participant_id | team_position | champion_id | champion_name | vision_score_per_min | cc_score_per_min | engage_success_rate | damage_taken_per_min |
|---|---|---|---|---|---|---|---|---|
| `1001` | `6` | `UTILITY` | `89` | Leona | `2.7` | `1.4` | `0.75` | `840.0` |
| `1001` | `7` | `UTILITY` | `111` | Nautilus | `2.3` | `1.7` | `0.71` | `910.0` |
| `1001` | `8` | `UTILITY` | `40` | Janna | `3.2` | `0.5` | `0.20` | `390.0` |
| `1001` | `3` | `MIDDLE` | `103` | Ahri | `1.1` | `0.2` | `0.05` | `510.0` |

### Resolved Matches

| participant_id | champion_name | matches `grp_support` | matches `grp_support_engage` | matches `grp_support_leona` |
|---|---|---|---|---|
| `6` | Leona | `yes` | `yes` | `yes` |
| `7` | Nautilus | `yes` | `yes` | `no` |
| `8` | Janna | `yes` | `no` | `no` |
| `3` | Ahri | `no` | `no` | `no` |

### Final Solution Table

| selected_group_id | source_group_id | matched_member_id | match_id | participant_id | champion_name | metric_id | metric_name | metric_value |
|---|---|---|---|---|---|---|---|---|
| `grp_support` | `grp_support` | `mem_support_any` | `1001` | `6` | Leona | `vision_score_per_min` | Vision Score Per Min | `2.7` |
| `grp_support` | `grp_support_engage` | `mem_leona_support` | `1001` | `6` | Leona | `vision_score_per_min` | Vision Score Per Min | `2.7` |
| `grp_support` | `grp_support_engage` | `mem_leona_support` | `1001` | `6` | Leona | `cc_score_per_min` | CC Score Per Min | `1.4` |
| `grp_support` | `grp_support_engage` | `mem_leona_support` | `1001` | `6` | Leona | `engage_success_rate` | Engage Success Rate | `0.75` |
| `grp_support` | `grp_support_leona` | `mem_leona_support` | `1001` | `6` | Leona | `vision_score_per_min` | Vision Score Per Min | `2.7` |
| `grp_support` | `grp_support_leona` | `mem_leona_support` | `1001` | `6` | Leona | `cc_score_per_min` | CC Score Per Min | `1.4` |
| `grp_support` | `grp_support_leona` | `mem_leona_support` | `1001` | `6` | Leona | `engage_success_rate` | Engage Success Rate | `0.75` |
| `grp_support` | `grp_support_leona` | `mem_leona_support` | `1001` | `6` | Leona | `damage_taken_per_min` | Damage Taken Per Min | `840.0` |
| `grp_support` | `grp_support` | `mem_support_any` | `1001` | `7` | Nautilus | `vision_score_per_min` | Vision Score Per Min | `2.3` |
| `grp_support` | `grp_support_engage` | `mem_nautilus_support` | `1001` | `7` | Nautilus | `vision_score_per_min` | Vision Score Per Min | `2.3` |
| `grp_support` | `grp_support_engage` | `mem_nautilus_support` | `1001` | `7` | Nautilus | `cc_score_per_min` | CC Score Per Min | `1.7` |
| `grp_support` | `grp_support_engage` | `mem_nautilus_support` | `1001` | `7` | Nautilus | `engage_success_rate` | Engage Success Rate | `0.71` |
| `grp_support` | `grp_support` | `mem_support_any` | `1001` | `8` | Janna | `vision_score_per_min` | Vision Score Per Min | `3.2` |

This is the correct first output because:

- the parent group stays visible
- the child branch stays visible
- the matched member stays visible
- inherited parent metrics remain available on deeper branches
- existing and built metrics share one shape
- the solution table can later be aggregated or pivoted

## Recommended First Milestone

The first correct milestone is:

"Given one selected group, Python resolves the full plan and ClickHouse builds one clean long-form solution table."

Once that is stable, everything else becomes much easier.

## Python Control Plane Template

The minimal control-plane template lives at:

- `app/worker/pipelines/dynamic_metrics_pipeline.py`

That single file should own:

- logical metadata types
- selected-group branch expansion
- member resolution
- parent-to-leaf metric inheritance
- metric dependency collection
- execution-plan construction

Start with one file first.

Only split it later if the real implementation becomes too large.
