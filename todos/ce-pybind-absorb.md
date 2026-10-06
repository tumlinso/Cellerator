

<!-- todo-orchestrator:v2-managed:start -->
# CE-PYBIND-ABSORB: Consolidate Cellerator Python and optional Torch bindings

Task revision: `7908`; current project revision is in `todo-status.md`.

## Objective
Implement the user-approved thin cellerator package and absorb CelleraTorch: one native owner, no compiler API, optional Torch build, preserve behavior/tests and actual consumers.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Implement the approved consolidation under this bounded claim; bind concrete acceptance gates once checks exist.

## Ownership
- `exclusive`: `AGENTS.md`
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `README.md`
- `exclusive`: `bench/ce_live/celleratorch`
- `exclusive`: `bench/ce_live/torch`
- `exclusive`: `bench/learning`
- `exclusive`: `bindings`
- `exclusive`: `cmake`
- `exclusive`: `compat/cp_math_v1/CMakeLists.txt`
- `exclusive`: `components/CelleraTorch`
- `exclusive`: `components/README.md`
- `exclusive`: `custom_torch_ops.md`
- `exclusive`: `docs/README.md`
- `exclusive`: `docs/architecture.qmd`
- `exclusive`: `docs/compiler/source-layout/split_umbrella_headers_and_public_component_imports.md`
- `exclusive`: `docs/current_implementation.qmd`
- `exclusive`: `docs/design/overview.md`
- `exclusive`: `docs/developer_reference.qmd`
- `exclusive`: `docs/development/python-bindings.md`
- `exclusive`: `docs/development/source-map.md`
- `exclusive`: `docs/development/start.md`
- `exclusive`: `docs/index.qmd`
- `exclusive`: `docs/learning`
- `exclusive`: `docs/migration_roadmap.qmd`
- `exclusive`: `docs/status/current.md`
- `exclusive`: `docs/storage_distribution_and_interop.qmd`
- `exclusive`: `examples/torch`
- `exclusive`: `experiments/moonshot-parallel-v1`
- `exclusive`: `experiments/moonshot-parallel-v1/diff/adapter`
- `exclusive`: `experiments/moonshot-parallel-v1/integration`
- `exclusive`: `include/Cellerator/bindings`
- `exclusive`: `include/Cellerator/compute/operation/model_ops`
- `exclusive`: `include/Cellerator/math/sampling`
- `exclusive`: `include/Cellerator/sampling.hh`
- `exclusive`: `out_of_scope_inventory.md`
- `exclusive`: `planning/python-bindings-absorb`
- `exclusive`: `planning/python-bindings-delivery`
- `exclusive`: `pyproject.toml`
- `exclusive`: `python/cellerator`
- `exclusive`: `scope.md`
- `exclusive`: `scripts/check_repository_layout.py`
- `exclusive`: `src/compute/operation/model_ops`
- `exclusive`: `src/compute/operation/product2/product2.cu`
- `exclusive`: `src/runtime/cuda_stream_device.cuh`
- `exclusive`: `src/runtime/relation_value_readiness.cu`
- `exclusive`: `tests/bindings`
- `exclusive`: `tests/ce_geo/run_build_matrix.py`
- `exclusive`: `tests/ce_geo/validation/run_full_volta_acceptance.py`
- `exclusive`: `tests/compiler/a04/split_umbrella_headers_and_public_component_imports_test.cc`
- `exclusive`: `tests/examples`
- `exclusive`: `tests/learning_consumer`
- `exclusive`: `tests/live/integration/wave_a_foundation_audit.py`
- `exclusive`: `tests/substrate/execution/README.md`
- `exclusive`: `tools/python_bindings`
- `forbidden`: `.todo-orchestrator`
- `forbidden`: `include/Cellerator/compiler`
- `forbidden`: `preprint`
- `forbidden`: `src/compiler`
- `forbidden`: `todo-status.md`
- `forbidden`: `todos`
- `forbidden`: `todos.md`
- `read`: `AGENTS.md`
- `read`: `components/CellShard`
- `read`: `include/Cellerator/compute/operation/indexed_mechanism`
- `read`: `include/Cellerator/compute/operation/product2`
- `read`: `include/Cellerator/compute/operators`
- `read`: `include/Cellerator/execution`
- `read`: `include/Cellerator/geometry`
- `read`: `include/Cellerator/parameters.hh`
- `read`: `include/Cellerator/quantized`
- `read`: `include/Cellerator/runtime`
- `read`: `src/compute/operation/indexed_mechanism`
- `read`: `src/compute/operation/product2`
- `read`: `src/compute/operators`
- `read`: `src/execution`
- `read`: `src/geometry`
- `read`: `src/quantized`
- `read`: `src/runtime`
- `read`: `tests/host`
- `read`: `tests/substrate`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
