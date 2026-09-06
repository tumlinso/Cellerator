# 7. Evidence, source authority and limitations

Cellerator: `b3340736c8c7b17266bd825c9f079d86e42a5639`, main worktree `wt-d3fea47b1244da16`. Project Control source: `9f4eca4efeada3b0bd3b05dc0ce3121a7099632c`, clean main worktree `wt-e8e946a37a2ee13e`. All reads were non-mutating. No live Cellerator build or CUDA test was run during package construction.

The source review remains in `basis/architecture-review.html`. Its historical observations must not be presented as fresh benchmarks. The two dirty library bodies could not be retrieved; a local content-preservation task is required. Todo authority unavailable means unknown state, not an empty task list.

## S01

`cellerator:planning/jbc-preledger-v1/08_DEPENDENCY_INTEGRATION_AND_PARALLELISM_MAP.md:1–99`

Serial queues within lanes; only genuine interface/checkpoint barriers; independent provider work.

## S02

`cellerator:planning/cellerator-compiler-preledger-v1/evidence/audit_accepted_todo_plan_schemas_and_live_precedents.md:1–134`

Historical installed native schema audit; v3 first-class runs/lanes and accepted field shapes. Not current live validation.

## S03

`cellerator:CMakeLists.txt:1–119`

Host SDK before CUDA-off return; native targets below; architecture70/toolchain settings.

## S04

`cellerator:include/Cellerator/execution/identity.hh:1–209`

Existing typed persistent identities, axis identity header, structure epoch and value generation.

## S05

`cellerator:src/compute/operation/builtin_catalog.cc:1–164`

Narrow actual builtin candidate/preparation regimes; N1 overlap and f16/f32 tuple.

## S06

`cellerator:src/compute/candidate/transpose_backward_candidate.cu:1–131`

Actual transpose CUDA kernel uses forward_value_positions and f16 weights/f32 arithmetic.

## S07

`cellerator:tests/math_core/transpose_backward_candidate_test.cu:1–249`

Concrete FMP1/CTP1 preparation fixture and source symbols to reuse. Source inspected, not test executed here.

## S08

`cellerator:src/runtime/value_readiness.cu:1–121`

Read in the authoritative review; generation/event mechanism to preserve, not blanket concurrent-reader proof.

## S09

`cellerator:src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc:1–156`

Existing relation IR lowering to v2, orientation/axis checks and self-referential lowered views.

## S10

`cellerator:src/compute/decomposition/support_embedding_v1.cc:1–80`

Contraction kind currently accepted only as disjoint channel concatenation, rejecting partial algebra.

## S11

`cellerator:include/Cellerator/compiler/sema/implement_operation_kind_resolution_v1.hh:1–38`

Representative primitive core field plus requires_composite_lowering boolean; seam for explicit classification.

## S12

`cellerator:examples/CMakeLists.txt:1–27`

Existing repository-local examples hierarchy.

## S13

`project-control:src/project_control/mutation.py:120–346`

Native validate+diff; coherent authority prerequisites; transaction checking; no automatic implementation.

## S14

`project-control:src/project_control/cli.py:95–240`

Exact local plan validate/apply --project --file forms; no --json on these parsers.

## S15

`project-control:docs/TOOL_CONTRACTS.md:174–260`

First-class workflow protocol versus subordinate delegate_task workers; native plan application boundaries.

## S16

`project-control:src/project_control/preledger.py:308–554`

Generic preledger compiler emits schema2, not the required first-class lane graph.

## S17

`project-control:src/project_control/services/planning.py:1–202`

Context reports [2] when workflow unavailable as a fallback, not a native-v3 rejection.

## S18

`project-control:src/project_control/admin.py:140–320`

Workspace materialization separate from plan ingestion; needs active run, clean base and managed integration coordination.

## S19

`cellerator:.todo-orchestrator/ce-ptr-plan.json:170–400, 860–899`

Native task/checkpoint/interface-consumer shapes and contract_split lane precedent.

## S20

`cellerator:tests/planner_targets.cmake:125–185`

Existing candidate test target names and real linked libraries.

## Primary external clarification

**R01: NVIDIA Floating Point and IEEE 754**

https://docs.nvidia.com/cuda/floating-point/index.html

FMA/association differences justify explicit policy and tolerance-based cross-device checks, not a blanket precision waiver.

**R02: CUDA Runtime API 12.9.1: Stream Management**

https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__STREAM.html

Stream ownership and explicit completion/readback reasoning.

**R03: CUDA Toolkit 13.0 Release Notes**

https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html

Volta offline build/library support removed in13; retain12.x toolchain for sm70.

## Confidence boundary

The proposed descriptor, pair API, task graph, target names and design choices are recommendations derived from these sources and the user’s instructions. They are not claims that new APIs already exist. Offline package validation is independent of the native Todo validator. Native acceptance/collision status is not available until a successful live preview. A reference-only fixture run checks only the supplied toy arithmetic. No performance advantage or complete compiler implementation is inferred.
