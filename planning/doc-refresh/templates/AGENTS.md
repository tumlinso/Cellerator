# Repository work contract

{{MANAGED_WORKFLOW_BLOCK}}

Read the [design](docs/design/overview.md) for intent, [current snapshot](docs/status/current.md) for its dated implementation boundary, and [source map](docs/development/source-map.md) for entry points. For an edit, the current scoped task and inspected source—not a historical plan or README completion sentence—determine what exists and what is authorized.

## Operate within the current authority

Use the installed Project Control/Codex front door and this repository's Todo authority. Take one bounded outcome, inspect only what it needs, preserve other claims/dirty work, and finish with actual evidence. Use configured subagents only for useful bounded work; the owner retains acceptance. Never edit SQLite, generated Todo views, recovery snapshots or generated context indexes by hand. Keep managed workflow blocks intact.

## Preserve the architecture while changing the implementation

{{PROJECT_AGENT_INVARIANTS}}

External compatibility is not an absolute constraint at this stage. A reviewed internal move may change names/interfaces if it improves development; repair real sibling consumers and relevant tests together. Frozen contracts and overlapping active work still require their owner's explicit reconciliation. Do not use a documentation task to rewrite numerical behavior, introduce another planner, or complete unrelated old epics.

## Validate what changed

Use the [development guide](docs/development/start.md) for the current commands. Run affected source/build/tests after code moves. For prose-only changes check links, status boundaries and rendered pages. Keep scientific claims scoped; record what was not run. Benchmarks require the existing assigned resources and clean timing interval. Include setup/transfer/routing when the claim needs them; no benchmark is launched by document generation.

Current work belongs in live status tools; experimental findings in [results](docs/results/index.md); durable rationale in design/development docs; superseded plans in the archive. Do not recreate the same mutable status table in all four places.
