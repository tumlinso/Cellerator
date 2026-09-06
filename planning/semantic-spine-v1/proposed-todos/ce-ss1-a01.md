# CE-SS1-A01: Separate scalar support dot from edge-channel product

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-ALGEBRA`. **Repository:** Cellerator.

## Objective

Remove the concrete contraction ambiguity without integrating the entire contraction portfolio.

## Prerequisites

`CE-SS1-I02`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.

- `src/compute/architecture/providers/nvidia/sm70/contract/sparse_contract_v1.cu`

## Exclusive write scope

- `include/Cellerator/compute/decomposition/support_embedding_v1.hh`
- `src/compute/decomposition/support_embedding_v1.cc`
- `include/Cellerator/compute/operation/support_product_semantics.hh`
- `tests/semantic_spine/algebra/support_product_test.cc`

## Implementation actions

1. Define scalar dot[e]=sum_k A[s(e),k]B[d(e),k] separately from product[e,k]=A[s(e),k]B[d(e),k]. Prefer an explicit result-shape/reduction tag rather than another arbitrary global enum migration.

2. Change the inspected support-embedding validator so a scalar dot cannot be accepted with concatenate-only assembly.

3. Retain the meaningful edge-channel decomposition under its truthful name/semantics. Existing scalar contraction kernels keep their dot-product meaning.

4. Reject legacy ambiguous descriptors until the caller specifies the missing meaning.

## Completion evidence

1. For K=2 and K=17, whole dot equals split-K partial sums; concatenating partial dots is rejected.

2. Edge-channel outputs concatenate disjoint K panels and retain their shape.

3. No contraction GPU provider is newly integrated or deleted.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
