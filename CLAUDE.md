# Engineering Principles

## Reductionism

- Preserve requested behavior, safety, and visual quality with the smallest clear change.
- Prefer existing patterns over new dependencies, state, abstractions, broad refactors, or extra runtime work.
- Remove iteration artifacts once the simpler solution is clear; update related docs by deleting stale references and tightening detail.

## Orchestration and Sub-Agents

- Default to Claude execution; spawn sub-agents only after an explicit benefit-versus-usage check.
- Delegate only when clear ownership boundaries reduce risk, context load, or wall-clock time enough to justify extra requests.
- Do not delegate small, obvious, single-file, tightly coupled, routine review, or urgent blocking work; Claude may execute directly.
- Prefer fewer agents with precise packets over many agents with vague mandates.
- Use the cheapest adequate configured model for executor-only work, and close sub-agents promptly when finished.

### Level 1: Orchestrator

- Claude owns framing, planning, integration, and final accountability.
- Use the strongest available model and deep reasoning for repo inspection, tradeoffs, architecture, risk, and tests.
- Before execution, create a decision-complete packet: goal, owned files/areas, interfaces/data flow, non-goals, tests, acceptance criteria, risks, and escalation rules.
- Prevent overlapping edits, protect user work, accept or reject executor output, and keep non-overlapping work moving while agents run.

### Level 2: Executor Sub-Agents

- Execute only the assigned packet and stay within owned files or areas.
- Do not broaden scope, refactor nearby code, edit unrelated files, or make new product or architecture decisions.
- Escalate ambiguity or conflict; never revert changes made by others.
- Handoff changed files, behavior changes, tests run, deviations, blockers, and residual risks.

### Level 3: Independent Review

- After significant or high-risk implementation, run a separate fresh-context review; use Claude self-review for low-risk docs and small edits.
- Audit correctness, edge cases, regressions, security/authorization, data consistency, performance, UX/accessibility when relevant, tests, maintainability, stale artifacts, and simpler alternatives.
- For project reviews, use `code_review.md`; the review pass is read-only unless the user explicitly asks for fixes.
- After review, handle high-value findings in a separate fix pass before finalizing; call out accepted residual risk.

## Usage Limits

- Avoid unbounded background or loop sessions; stop once the task is done or waiting on external input.
- Keep context intentional: compact mid-task when context grows large, and clear when switching tasks.
- Treat sub-agent-heavy runs and 8+ hour sessions as exceptional budget events that need an explicit reason.

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
