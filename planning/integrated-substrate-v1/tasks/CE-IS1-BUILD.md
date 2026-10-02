# CE-IS1-BUILD — Make components installable without dependency cycles

## Outcome

Design and test granular CMake/package boundaries and installed-consumer fixtures for the mathematical library and optional adapters.

## Implementation direction

Torch is optional for core math. Pin a verified CUDA 12.x SM70 toolchain rather than assuming a CUDA 13 upgrade is safe. Keep existing package/component exports or rename them deliberately with sibling repairs.

## Dependencies and ownership

Local predecessors: CE-IS1-ADOPT.

Writable roots: `cmake/substrate`, `tests/substrate/install`, `examples/substrate/install`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- CPU/reference or compiler components and SM70 numerical components are independently selectable where source permits.
- Baseplane exact-core ↔ CE bridge ↔ BP representation dependency graph is acyclic at target level.
- No supported consumer compiles sibling private source; central build patch is returned for MERGE-A.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
