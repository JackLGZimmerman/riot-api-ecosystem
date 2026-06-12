# Project Review Prompt

This review is read-only. Inspect the diff, relevant files, and available handoff context, but do not edit files, stage, commit, push, format, run migrations, generate code, or run commands that may mutate repo state.

Lead with findings ordered by severity. Include file and line references where possible. If there are no findings, say so and note any residual uncertainty.

## Orchestration Check

- Confirm the implementation was established by a Level 1 orchestrator before execution began.
- Confirm any executor sub-agents only executed a decision-complete packet after the plan existed.
- Confirm the orchestrator used the highest-level available model for framing, architecture, risk, and acceptance criteria.
- Confirm executor sub-agents, if used, were lower/cheaper/specialized models appropriate for bounded execution.
- Flag missing or weak evidence as `Unverified` unless the diff itself proves a violation.
- Flag as a finding if a sub-agent made product or architecture decisions, broadened scope, edited outside ownership, or executed before the implementation packet was established.
- If no sub-agents were used, confirm direct top-level execution was appropriate for the scope.

## Code and Behavior Review

Audit correctness, edge cases, regressions, security/authorization, data consistency, performance, UX/accessibility when relevant, tests, maintainability, stale artifacts, and simpler alternatives.

## Output Shape

- `Findings`: actionable issues first, or `None`.
- `Orchestration`: `Pass`, `Fail`, or `Unverified`, with concise evidence.
- `Tests`: checks observed or recommended.
- `Residual Risk`: anything important that remains uncertain.
