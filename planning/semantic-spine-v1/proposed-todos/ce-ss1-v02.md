# CE-SS1-V02: Build contract and identity regression probes

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-VERIFY`. **Repository:** Cellerator.

## Objective

Exercise semantic invariants that a tiny numerical example alone can miss.

## Prerequisites

`CE-SS1-V01`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `tests/semantic_spine/verification/contract_probes.cc`
- `tests/semantic_spine/verification/CMakeLists.txt`

## Implementation actions

1. Check high-bit-only identity changes, source/destination axis mistakes at equal extents, stale epochs/order, pointer changes and semantic equality across provenance.

2. Test exact support permutation recovery without requiring a new layout optimizer.

3. Cover invalid input capacities and 64-bit cardinality/local-index boundaries using metadata-only rejection tests, not enormous allocations.

4. Do not couple these tests to native or frontend implementation source; both later import the same independent fixtures.

## Completion evidence

1. Malformed descriptors fail before access; valid-but-unsupported physical sizes return an explicit status.

2. Inputs/outputs cannot alias in the supported execution regime.

3. Checks operate on values/contracts rather than searching source for desired words.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
