# Engineering Principles

## Reductionism

- Preserve requested behavior, safety, and visual quality with the smallest clear change.
- Prefer existing patterns over new dependencies, state, abstractions, broad refactors, or extra runtime work.
- Remove iteration artifacts once the simpler solution is clear; update related docs by deleting stale references and tightening detail.

## Orchestration and Sub-Agents

- Default to Claude execution. Treat sub-agents as an explicit cost and risk tradeoff, not a habit.
- Delegate only when isolated ownership, fresh context, specialization, or parallelism clearly justifies the extra requests.
- Do not delegate small, obvious, single-file, tightly coupled, routine review, urgent blocking, or run-only background work.
- Before spawning, prepare a decision-complete packet: goal, owned files/areas, interfaces/data flow, non-goals, tests, acceptance criteria, risks, escalation rules, and resource limits.
- Prefer one or two precise executor/reviewer agents over broad agent trees; use the cheapest adequate configured model and close agents promptly.
- Executor sub-agents must not create nested sub-agent loops unless Claude explicitly authorized that shape.

### Level 1: Orchestrator

- Claude owns framing, planning, integration, and final accountability.
- Use the strongest available model and deep reasoning for repo inspection, tradeoffs, architecture, risk, and tests.
- Prevent overlapping edits, protect user work, accept or reject executor output, and keep non-overlapping work moving while agents run.

### Level 2: Executor Sub-Agents

- Execute only the assigned packet and stay within owned files or areas.
- Do not broaden scope, refactor nearby code, edit unrelated files, or make new product or architecture decisions.
- Escalate ambiguity or conflict; never revert changes made by others.
- Handoff changed files, behavior changes, tests run, deviations, blockers, and residual risks.

### Level 3: Independent Review

- After significant or high-risk implementation, run a separate fresh-context review only when the added requests are justified; use Claude self-review for low-risk docs and small edits.
- Audit correctness, edge cases, regressions, security/authorization, data consistency, performance, UX/accessibility when relevant, tests, maintainability, stale artifacts, and simpler alternatives.
- For project reviews, use `code_review.md`; the review pass is read-only unless the user explicitly asks for fixes.
- After review, handle high-value findings in a separate fix pass before finalizing; call out accepted residual risk.

## Dynamic and Loop Work

- Before loops, sweeps, polling, background jobs, or repeated retries, define the command, resource limits, artifact/log path, stop condition, and escalation point.
- Serialize GPU/cache-heavy or RAM-heavy work. For ML reviews or experiment loops, check `nvidia-smi` and `free -h`; avoid pytest, `app.ml` imports, ClickHouse raw aggregations, and cache-loading scripts while training is active unless explicitly intended.
- Keep run-only/background jobs orchestrator-owned unless a bounded packet explicitly delegates monitoring.

## Usage Limits

- Avoid unbounded background or loop sessions; stop once the task is done or waiting on external input.
- Keep context intentional: compact mid-task when context grows large, and clear when switching tasks.
- Treat sub-agent-heavy runs, long-running loops/background jobs, and 8+ hour sessions as exceptional budget events that need an explicit reason and stop condition.

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
