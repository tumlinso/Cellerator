# CE-SS1-C03: Specify provisional native pair preparation and bindings

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-CORE`. **Repository:** Cellerator.

## Objective

Give parallel consumers a small native execution contract for one prepared forward/transpose pair, without freezing the eventual SDK.

## Prerequisites

`CE-SS1-C02`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.



## Exclusive write scope

- `include/Cellerator/compute/operation/prepared_relation.hh`
- `docs/semantic_spine_v1/native_contract.md`

## Implementation actions

1. Define a declaration-only prepared pair interface with caller stream and borrowed device inputs/outputs, structured errors, topology lifetime, explicit refresh and destruction.

2. Initially support N1, f16 relation storage and f32 state/multiply/accumulation/output using existing kernels. No Tensor Core requirement and no CPU success fallback in accelerator acceptance.

3. Make value publication monotonic per pair and stream-ordered; distinguish latest enqueued generation from device completion. Pair use is single-device/single-stream in v1 and rejects conflicting stream/device.

4. Allow host topology preparation, allocate required projection/packing capacity before enqueue, and expose actual preparation/refresh/launch evidence. Define same-generation behavior, failure-state preservation and required caller fences.

## Completion evidence

1. Every demo call has a declared contract and explicit failure behavior; no implementation or fake backend is supplied by this task.

2. Host-only semantic headers do not include CUDA merely to express an operation.

3. The native interface has no hidden hot-path allocation, geometry discovery or canonicalization promise.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
