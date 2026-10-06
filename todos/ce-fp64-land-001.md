

<!-- todo-orchestrator:v2-managed:start -->
# CE-FP64-LAND-001: Land qualified FP64 with integrated bindings and moments

Task revision: `7942`; current project revision is in `todo-status.md`.

## Objective
Finish the user-requested FP64 integration on canonical main: preserve existing binding/moments commits and unrelated changes, make qualification portable, rebuild combined native and adapter consumers, commit scoped FP64 sources and records, push and verify HEAD equals origin/main.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Claim landing task, delegate portable qualification and combined checks; root retains exact staging, integration acceptance and publication.

## Ownership
- `exclusive`: `docs/development/fp64.md`
- `exclusive`: `docs/results/fp64-qualification.md`
- `exclusive`: `include/Cellerator/compute/candidate/sparse/project.hh`
- `exclusive`: `include/Cellerator/compute/operation/device_elementwise.hh`
- `exclusive`: `include/Cellerator/compute/operation/prepared_relation.hh`
- `exclusive`: `include/Cellerator/compute/operation/relation_semantics.hh`
- `exclusive`: `planning/fp64`
- `exclusive`: `src/compute/candidate/sparse/project.cu`
- `exclusive`: `src/compute/operation/device_elementwise.cuh`
- `exclusive`: `src/compute/operation/prepared_relation.cu`
- `exclusive`: `src/compute/operation/relation_semantics.cc`
- `exclusive`: `tests/fp64`
- `forbidden`: `.todo-orchestrator`
- `forbidden`: `CMakeLists.txt`
- `forbidden`: `bindings`
- `forbidden`: `cmake`
- `forbidden`: `components/CelleraTorch`
- `forbidden`: `include/Cellerator/compiler`
- `forbidden`: `include/Cellerator/compute/operation/native_numeric`
- `forbidden`: `python`
- `forbidden`: `src/compiler`
- `forbidden`: `src/compute/operation/native_numeric`
- `forbidden`: `todo-status.md`
- `forbidden`: `todos`
- `forbidden`: `todos.md`
- `read`: `AGENTS.md`
- `read`: `CMakeLists.txt`
- `read`: `bindings`
- `read`: `cmake`
- `read`: `docs/architecture.qmd`
- `read`: `docs/current_implementation.qmd`
- `read`: `docs/developer_reference.qmd`
- `read`: `docs/development/python-bindings.md`
- `read`: `examples/native_neighborhood_moments`
- `read`: `examples/semantic_spine_v1`
- `read`: `include/Cellerator`
- `read`: `planning/python-bindings-absorb`
- `read`: `python`
- `read`: `src/compute`
- `read`: `tests`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
