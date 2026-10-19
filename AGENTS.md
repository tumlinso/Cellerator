# Repository work contract

<!-- project-control:start -->
## Project Control and native Todo workflow

Use Project Control's `overview`, `search`, `delta`, `frontier`, `history`,
`evidence`, and `impact` tools for registered-workspace context. They are not
the Todo transaction API. For Todo state and workflow, use the native `todo_*`
tools exposed by the host. Start with `todo_read_authority`; for plan changes,
call `todo_preview_plan`, inspect the returned preview, then pass that exact
preview to `todo_commit_plan` with an idempotency key. Use the corresponding
`todo_open_run`, `todo_create_workspace`, `todo_dispatch_attempt`, outcome,
verification, and recovery tools for later lifecycle steps.

Arguments use a registered workspace ID and, when necessary, an optional
registered `repository` alias. The host binds authority, identity, credentials,
profile, and providers; never pass filesystem roots or bearer credentials as
selectors. Read migrated history with `todo_search_legacy_history`, current
status with `todo_read_legacy_current_tasks`, and preserved declarations with
`todo_read_legacy_project_context`. Imported completion is a source assertion,
not native verification; unknown legacy effects require attention and are never
replayed automatically. Project Control's existing non-Todo tools remain
available for their own purposes.
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

## Hardware and algorithm choice

Use the [CUDA skill](/home/tumlinson/.agents/skills/cuda/SKILL.md) for GPU
computation, porting, correctness, profiling and optimization. Follow its focused
guidance for the actual operation and target hardware, including host execution
rules when running GPU work.

Aim to make full use of the hardware. Reflect on algorithm, representation and
ownership choices before tuning a kernel: consider how layout, locality, reuse,
fusion and precision can make the semantic operation native to the machine.
Consider unconventional algorithms or encodings when they offer a sound fit.
Choose wisely using an independent correctness oracle and complete workload
costs, including preparation, conversion, memory and communication. Compare
with the strongest conventional or library alternative and retain the evidence
that supports or rejects each candidate.

## Software ports and design probes

During active library development, any explicitly user-authorized software
port or design probe with a scoped Project Control task carries standing
permission for additive changes to Cellerator, without renewed permission. Keep
additions general, clean, and maintainable, with explicit contracts and
validation; use small architectural cleanup only to resolve demonstrated
friction. General numerical, state, graph, relation, statistical, and
iterative mechanisms belong in Cellerator; Baseplane should own only
intrinsically sequence-grounded functionality.

Use FP16 as the general computation default, retaining FP32, FP64 and other
supported precision types as explicit alternatives. FP16 execution may use
ordinary CUDA arithmetic or Tensor Cores; choose the implementation by the
operation's contract and measured complete cost. Declare storage, multiply,
accumulation and output types separately, using FP32 accumulation where
appropriate. Respect explicit caller precision and existing operation contracts.
Compare approximate computation with upstream/native computation; a
higher-precision reference may supplement that comparison. Keep numerical claims
within demonstrated evidence and retain suitable alternatives for unsupported
or unsafe regimes. See [Numerical policy](docs/architecture.qmd#numerical-policy)
for the provisional evidence boundary and current implementation distinction.

Prefer templates for algorithms whose behavior is independent of numeric type.
Keep genuinely type-specific arithmetic, numerical constraints and hardware
specializations explicit; do not force them into a shared implementation.

When a basic reusable primitive appears missing or insufficient, report this
promptly to the user. Include the concrete computation and evidence, current
library capability and gap, proposed owner and general contract, and
downstream impact. Discuss major API, granularity, ownership, or architectural
choices with the user before committing to them.

## Validate what changed

Use the [developer reference](docs/developer_reference.qmd) for build and test commands. Run affected source/build/tests after code moves. For prose-only changes check links, status boundaries and rendered pages. Keep scientific claims scoped; record what was not run. Benchmarks require the existing assigned resources and clean timing interval. Include setup/transfer/routing when the claim needs them; no benchmark is launched by document generation.

Current work belongs in live status tools; experimental findings stay with their source-bound records and the dated [implementation record](docs/current_implementation.qmd); durable rationale belongs in design/development docs; superseded plans stay in the archive. Do not recreate the same mutable status table in all four places.
