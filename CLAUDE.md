# Engineering Principles

## Engineering Reductionism

- Simplify every code change as much as possible while preserving the requested behavior, safety, and visual quality.
- Prefer the smallest correct implementation over additive architecture, extra state, new abstractions, or broad refactors.
- Apply a reductionist mindset before editing: ask whether the same outcome can be achieved with fewer moving parts, fewer lines of code, less duplicated logic, and less runtime work.
- If less code can deliver the same functionality with equal clarity and maintainability, use less code.
- Keep changes efficient in both implementation and runtime: avoid unnecessary dependencies, observers, effects, re-renders, network calls, database work, and layout measurement.
- Remove stale or redundant code created during iteration once the simpler solution is clear.
- Related documentation (`.md`) should be updated, stale references removed, and sections rewritten when leaner detail would be clearer.

## Orchestration and Sub-Agent Discipline

- Use sub-agents only when the task has clear ownership boundaries that make delegation reduce risk, context load, or wall-clock time.
- Do not delegate small, obvious, single-file, tightly coupled, or urgent blocking work. Claude may execute directly when delegation would add ceremony or risk.
- Prefer fewer sub-agents with sharper task packets over many agents with fuzzy mandates.

### Level 1: Orchestrator

- Claude, as the top-level orchestration model, owns strategy, integration, and accountability for the whole change.
- Use the strongest available top-level model and deep reasoning for problem framing, repo inspection, risk analysis, architecture decisions, and implementation planning.
- Before delegating execution, produce a decision-complete implementation packet that includes:
  - Goal and success criteria.
  - Files, modules, or responsibilities owned by the executor.
  - Interfaces, data flow, and compatibility constraints.
  - Non-goals and boundaries.
  - Tests and acceptance criteria.
  - Known risks, edge cases, and escalation rules.
- Keep integration authority at the orchestration level: resolve conflicts, prevent overlapping edits, protect user changes, and decide whether executor output is accepted.
- Claude should continue meaningful non-overlapping work while sub-agents run instead of idly waiting.

### Level 2: Executor Sub-Agents

- Executor sub-agents implement only the packet they were assigned.
- Executors must stay inside their ownership area, avoid unrelated files, avoid broad refactors, and avoid improving nearby code unless the packet explicitly requires it.
- Executors must escalate ambiguity, missing context, or scope pressure instead of inventing new product or architecture decisions.
- Executors must assume other agents or the user may be editing the codebase at the same time and must not revert changes they did not make.
- Executor handoff must report changed files, behavior changes, tests run, deviations from the packet, blockers, and residual risks.

### Level 3: Independent Review

- After every significant implementation, run a separate review agent with a fresh, independent context.
- The review must audit functional correctness, edge cases, regressions, security and authorization risks, performance costs, accessibility and UX clarity when relevant, data consistency, error handling, test coverage, maintainability, stale artifacts, and whether the same outcome can be achieved more simply.
- Treat review findings as actionable input. Fix high-value findings before finalizing, and explicitly call out any residual risk that remains.
- The review agent critiques the integrated result; it does not replace Claude's accountability.

## User Work Safety

- Check `git status` before planning edits, before integration, and before any commit.
- Treat untracked files and unexpected modifications as user-owned unless proven otherwise.
- Never revert, overwrite, stage, format, or commit unrelated user changes.
- If assigned files contain user edits, read them first and integrate around them. Ask only if the conflict cannot be resolved safely.

## Commit Discipline

- After every significant completed change, stage, commit, and push the work when the repository is ready.
- Keep commits scoped to the completed change unless the user explicitly asks to stage everything with `git add .`.
- Before committing, check the working tree and avoid unintentionally including unrelated user changes.
- Use clear commit messages that describe the behavior or fix, not implementation noise.

## Memory System

- Before planning, check relevant summaries in `/home/jack/.codex/memory/` and apply lessons that match the current repo or task.
- Store one lesson per file with a one-line summary at the top.
- Record corrections and confirmed approaches alike, including why they mattered.
- Do not save what the repo or chat history already records.
- Update an existing note rather than creating a duplicate; delete notes that turn out to be wrong.
- For `riot-api-ecosystem`, keep known operational lessons visible in planning: backup-first and reversible matchdata repairs, Riot continent versus platform-region distinctions, restart automation safety, service dependency checks, and model-serving interface boundaries.
