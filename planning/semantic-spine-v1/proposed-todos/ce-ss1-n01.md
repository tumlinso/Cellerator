# CE-SS1-N01: Adapt the canonical descriptor to existing preparation

**Status:** proposed; not applied or implemented.

**First-class lane:** `CE-SS1-L-NATIVE`. **Repository:** Cellerator.

## Objective

Translate canonical semantics into the narrow existing executable candidate contracts without dropping meaning.

## Prerequisites

`CE-SS1-I02`

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, the lane playbook, and the task-specific sources below. The review is the design basis; current source is evidence.

- `src/compute/operation/builtin_catalog.cc`
- `src/compute/operation/preparation_factory.cc`
- `src/compute/candidate/feature_major_small_n_candidate.cu`
- `src/compute/candidate/transpose_backward_candidate.cu`

## Exclusive write scope

- `src/compute/operation/prepared_relation.cu`
- `tests/semantic_spine/native/adapter_test.cu`

## Implementation actions

1. Write one checked adapter for the FMP1 N1 forward and CTP1 N1 transpose path. Compare numeric types separately: f16 weights must not be overwritten by f32 state metadata.

2. Use actual candidate/preparation entry points. Retain the canonical descriptor for validation; temporary legacy views are mechanism adapters, not a second authority.

3. Reject unsupported width/numeric/update/alias policy before any launch, preserving output and pair state. Do not reinterpret a composition as an apply.

4. Reuse useful legacy code; limit modifications to adapter ownership unless a demonstrated bug requires an explicitly scoped integration edit.

## Completion evidence

1. Descriptor fields round-trip through the adapter without low/high-bit loss or swapped biological identities.

2. A valid but unsupported request returns a capability error instead of a success stub or fallback CPU calculation.

Record the actual command, exit status, test cases, source revision and artifact hashes. A planned assertion, file existence or name-bearing receipt is not a passed test. Keep runtime/GPU evidence distinct from metadata-only checks.

## Scope limit and failure policy

Do not broaden the operation portfolio or rewrite a working backend to make an adapter easier. No full language/SDK/JIT/planner expansion. No unrelated branch or Todo cleanup. Block or hand off the specific issue rather than silently weakening tests.

## Coordination

Use `next_task`, `inspect_task` and `coordinate_task` through the installed first-class workflow. The coordinator must bind the subagent to this exact run/lane. Do not use `delegate_task` to impersonate a first-class lane. Commit after the task, preserve evidence, and hand off only a real source artifact. Pause immediately on the user’s instruction. Shared contract revisions require notifying every consumer before resumption.
