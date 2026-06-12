# Engineering Principles

## Reductionism

- Preserve requested behavior, safety, and visual quality with the smallest clear change.
- Prefer existing patterns over new dependencies, state, abstractions, broad refactors, or extra runtime work.
- Remove iteration artifacts once the simpler solution is clear; update related docs by deleting stale references and tightening detail.

## Orchestration and Sub-Agents

- Delegate only when clear ownership boundaries reduce risk, context load, or wall-clock time.
- Do not delegate small, obvious, single-file, tightly coupled, or urgent blocking work; the top-level agent may execute directly.
- Prefer fewer agents with precise packets over many agents with vague mandates.

### Level 1: Orchestrator

- The top-level model owns framing, planning, integration, and final accountability.
- Use the strongest available model and deep reasoning for repo inspection, tradeoffs, architecture, risk, and tests.
- Before execution, create a decision-complete packet: goal, owned files/areas, interfaces/data flow, non-goals, tests, acceptance criteria, risks, and escalation rules.
- Prevent overlapping edits, protect user work, accept or reject executor output, and keep non-overlapping work moving while agents run.

### Level 2: Executor Sub-Agents

- Execute only the assigned packet and stay within owned files or areas.
- Do not broaden scope, refactor nearby code, edit unrelated files, or make new product or architecture decisions.
- Escalate ambiguity or conflict; never revert changes made by others.
- Handoff changed files, behavior changes, tests run, deviations, blockers, and residual risks.

### Level 3: Independent Review

- After significant implementation, run a separate fresh-context review.
- Audit correctness, edge cases, regressions, security/authorization, data consistency, performance, UX/accessibility when relevant, tests, maintainability, stale artifacts, and simpler alternatives.
- Fix high-value findings before finalizing; call out accepted residual risk.

## User Work Safety

- Check `git status` before edits, integration, and commit.
- Treat untracked or unexpected changes as user-owned unless proven otherwise.
- Never revert, overwrite, stage, format, or commit unrelated user changes.
- If target files contain user edits, read first and integrate around them; ask only when conflict cannot be resolved safely.

## Commit Discipline

- Stage, commit, and push significant completed changes when the repo is ready.
- Stage only completed scope unless the user explicitly asks for `git add .`.
- Use behavior-focused commit messages.

## Memory

- Before planning, scan relevant summaries in `/home/jack/.codex/memory/`.
- Keep one lesson per file with a one-line summary; update or delete existing notes rather than duplicating stale guidance.
- Record only durable corrections or confirmed approaches and why they matter.
- In this repo, keep known risks in view: reversible matchdata repairs, continent/platform distinctions, restart automation, service dependencies, and model-serving boundaries.
