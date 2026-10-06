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

Read the [architecture](docs/architecture.qmd) for intent, [current implementation record](docs/current_implementation.qmd) for its dated boundary, and [developer reference](docs/developer_reference.qmd) for entry points. The [Python binding guide](docs/development/python-bindings.md) covers the current package surface. For an edit, the current scoped task and inspected source—not a historical plan or README completion sentence—determine what exists and what is authorized.

## Operate within the current authority

Use the installed Project Control/Codex front door and this repository's Todo authority. Take one bounded outcome, inspect only what it needs, preserve other claims/dirty work, and finish with actual evidence. Use configured subagents only for useful bounded work; the owner retains acceptance. Never edit SQLite, generated Todo views, recovery snapshots or generated context indexes by hand. Keep managed workflow blocks intact.

## Preserve the architecture while changing the implementation

- Preserve biological domain/order identity and distinguish immutable structure epochs from mutable value generations.
- Keep discovery/semantic geometry separate from physical realization and actual runtime state. Prediscovered runtime gates are part of the intended design; do not relabel a narrower value update as complete adaptive execution.
- Keep execution order/conversion, allocation, streams and complete costs explicit. Candidate choice is measured, not mandated by a favored sparse format or Tensor Core path.
- Cellerator owns general compilation and numerical execution, including its optional Python and Torch adapters. Baseplane owns sequence meaning, GH owns scientific inference, and CellShard owns persistence/delivery. Preserve actual cross-project consumers.
- Existing code/contract evidence survives cleanup. Resolve a real interface change through its owner; no silent broad ABI rewrite.

External compatibility is not an absolute constraint at this stage. A reviewed internal move may change names/interfaces if it improves development; repair real sibling consumers and relevant tests together. Frozen contracts and overlapping active work still require their owner's explicit reconciliation. Do not use a documentation task to rewrite numerical behavior, introduce another planner, or complete unrelated old epics.

## Software ports and design probes

During active library development, any explicitly user-authorized software
port or design probe with a scoped Project Control task carries standing
permission for additive changes to Cellerator, without renewed permission. Keep
additions general, clean, and maintainable, with explicit contracts and
validation; use small architectural cleanup only to resolve demonstrated
friction. General numerical, state, graph, relation, statistical, and
iterative mechanisms belong in Cellerator; Baseplane should own only
intrinsically sequence-grounded functionality.

Preserve native computation by default. An optional, opportunistic FP16 Tensor
Core mode is the only permitted deliberate numerical deviation. Compare it
with upstream/native computation. A higher-precision reference may supplement
that comparison. Prefer FP32 accumulation where supported, and declare and
qualify actual accumulation and output policy. Use the mode only within its
demonstrated numerical envelope, retaining native fallback for unsupported or
unsafe regimes. Existing Tensor Core documentation defines implementation
requirements.

When a basic reusable primitive appears missing or insufficient, report this
promptly to the user. Include the concrete computation and evidence, current
library capability and gap, proposed owner and general contract, and
downstream impact. Discuss major API, granularity, ownership, or architectural
choices with the user before committing to them.

## Validate what changed

Use the [developer reference](docs/developer_reference.qmd) for build and test commands. Run affected source/build/tests after code moves. For prose-only changes check links, status boundaries and rendered pages. Keep scientific claims scoped; record what was not run. Benchmarks require the existing assigned resources and clean timing interval. Include setup/transfer/routing when the claim needs them; no benchmark is launched by document generation.

Current work belongs in live status tools; experimental findings stay with their source-bound records and the dated [implementation record](docs/current_implementation.qmd); durable rationale belongs in design/development docs; superseded plans stay in the archive. Do not recreate the same mutable status table in all four places.
