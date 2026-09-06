# CE-SS1-C01: Define one value-owned relation application descriptor

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-CORE`. **Repository:** Cellerator.

## Objective

Create the smallest canonical mathematical descriptor for exact forward/transpose application, reusing existing biological identities rather than canonizing a versioned operation enum.

## Prerequisites

`CE-SS1-I01`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.

- `include/Cellerator/execution/identity.hh`
- `include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh`

## Exclusive write scope

- `include/Cellerator/compute/operation/relation_semantics.hh`
- `docs/semantic_spine_v1/contract.md`

## Implementation actions

1. Use the supplied declaration sketch as a concrete starting point, not an ABI commitment. Keep source provenance, live buffers, stream, candidate, projections and mutable generations outside mathematical identity.

2. Retain typed source/destination domain-and-order identities, extents, topology ID/epoch, edge order/count, orientation and separate relation/input/multiply/accumulation/output types.

3. Define each logical edge as an additive contribution, including distinct duplicate endpoint edges. Empty rows sum to zero; transpose reverses traversal, not biological identities.

4. Specify overwrite and recognized-but-unsupported update/alias modes honestly. Describe topology reuse and exact support separately from floating-point equivalence.

## Completion evidence

1. A minimal C++ compile verifies the descriptor is value-owned without self-referential pointers.

2. A non-square asymmetric fixture determines unambiguous input/result axes for both orientations.

3. No new generic graph system, blanket v2 migration, or physics-dependent identity is introduced.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
