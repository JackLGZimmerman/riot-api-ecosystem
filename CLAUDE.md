# Engineering Reductionism

- Simplify every code change as much as possible while preserving the requested behavior, safety, and visual quality.
- Prefer the smallest correct implementation over additive architecture, extra state, new abstractions, or broad refactors.
- Apply a reductionist mindset before editing: ask whether the same outcome can be achieved with fewer moving parts, fewer lines of code, less duplicated logic, and less runtime work.
- If less code can deliver the same functionality with equal clarity and maintainability, use less code.
- Keep changes efficient in both implementation and runtime: avoid unnecessary dependencies, observers, effects, re-renders, network calls, database work, and layout measurement.
- Remove stale or redundant code created during iteration once the simpler solution is clear.

# Commit Discipline

- After every significant completed change, stage, commit, and push the work when the repository is ready.
- Keep commits scoped to the completed change unless the user explicitly asks to stage everything with `git add .`.
- Before committing, check the working tree and avoid unintentionally including unrelated user changes.
- Use clear commit messages that describe the behavior or fix, not the implementation noise.

# Sub-Agent Review Discipline

- Use sub-agents whenever a task has isolated parts of the functionality that can be implemented, researched, or reviewed independently.
- Split sub-agent work by meaningful boundaries such as client UI, server/API behavior, data modeling, authorization, performance, tests, accessibility, and visual polish.
- After every significant adjustment, run a separate review agent to perform a thorough audit of the change.
- The review audit should look beyond simple bugs: verify functional correctness, edge cases, regressions, security and authorization risks, performance costs, accessibility, UX clarity, data consistency, error handling, test coverage, maintainability, redundant code, stale artifacts, and whether the same outcome could be achieved more simply.
- Treat review findings as actionable input: fix high-value findings before finalizing, and explicitly call out any residual risk that remains.
