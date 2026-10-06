# Repository work contract

<!-- project-control:start -->
## Project Control workflow

For substantial repository work, use `project-control`. Start with `next_task`,
use `inspect_task` for bounded current-task context, and use `coordinate_task`
for typed synchronization. Rich Project Control reads are secondary escalation
tools when bounded workflow context is insufficient.

Todo Orchestrator remains the transactional authority. First-class Codex agents
receive lanes and roles. Use configured Codex subagents for bounded research,
implementation, tests and review under the root claim. Local workers are reserved
for Project Control observers for now; do not use `delegate_task` or
`local-coding-worker` for implementation delegation. Subagents do not take over
the root's task lifecycle or final acceptance.
<!-- project-control:end -->

Read the [design](docs/design/overview.md) for intent, [current snapshot](docs/status/current.md) for its dated implementation boundary, and [source map](docs/development/source-map.md) for entry points. For an edit, the current scoped task and inspected source—not a historical plan or README completion sentence—determine what exists and what is authorized.

## Operate within the current authority

Use the installed Project Control/Codex front door and this repository's Todo authority. Take one bounded outcome, inspect only what it needs, preserve other claims/dirty work, and finish with actual evidence. Use configured subagents only for useful bounded work; the owner retains acceptance. Never edit SQLite, generated Todo views, recovery snapshots or generated context indexes by hand. Keep managed workflow blocks intact.

## Preserve the architecture while changing the implementation

- Preserve biological domain/order identity and distinguish immutable structure epochs from mutable value generations.
- Keep discovery/semantic geometry separate from physical realization and actual runtime state. Prediscovered runtime gates are part of the intended design; do not relabel a narrower value update as complete adaptive execution.
- Keep execution order/conversion, allocation, streams and complete costs explicit. Candidate choice is measured, not mandated by a favored sparse format or Tensor Core path.
- Cellerator owns general compilation/numerical execution. Baseplane owns sequence meaning, GH owns scientific inference, CelleraTorch is an adapter and CellShard owns persistence/delivery. Preserve actual cross-project consumers.
- Existing code/contract evidence survives cleanup. Resolve a real interface change through its owner; no silent broad ABI rewrite.

External compatibility is not an absolute constraint at this stage. A reviewed internal move may change names/interfaces if it improves development; repair real sibling consumers and relevant tests together. Frozen contracts and overlapping active work still require their owner's explicit reconciliation. Do not use a documentation task to rewrite numerical behavior, introduce another planner, or complete unrelated old epics.

## scVelo and CellRank library probes

During a user-authorized scVelo or CellRank probe with a scoped Project Control
task, additive Cellerator changes are permitted without renewed permission.
Keep additions general, clean, and maintainable, with explicit contracts and
validation; use small architectural cleanup only to resolve demonstrated
friction. Preserve native scientific computation by default. An optional,
opportunistic FP16 Tensor Core mode is the only permitted deliberate numerical
deviation. Compare it with upstream/native computation; a higher-precision
reference may supplement that comparison. Prefer FP32 accumulation where
supported, and declare and qualify actual accumulation and output policy. Use
the mode only within its demonstrated numerical envelope, retaining native
fallback for unsupported or unsafe regimes. Existing Tensor Core documentation
defines implementation requirements.
Report missing or insufficient reusable primitives promptly with the concrete
computation, evidence, existing capability and gap, proposed owner and general
contract, and downstream impact. Discuss major API, granularity, or ownership
choices with the user before committing to them.

## Validate what changed

Use the [development guide](docs/development/start.md) for the current commands. Run affected source/build/tests after code moves. For prose-only changes check links, status boundaries and rendered pages. Keep scientific claims scoped; record what was not run. Benchmarks require the existing assigned resources and clean timing interval. Include setup/transfer/routing when the claim needs them; no benchmark is launched by document generation.

Current work belongs in live status tools; experimental findings in [results](docs/results/index.md); durable rationale in design/development docs; superseded plans in the archive. Do not recreate the same mutable status table in all four places.
