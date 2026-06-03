# Out-Of-Scope Migration Inventory

Last updated: 2026-05-04

This advisory ledger marks surfaces that are currently outside the durable
Cellerator scope in `scope.md`, or are only temporarily inside Cellerator during
migration. It does not fail builds. Keep it agent-readable: every entry names a
destination owner, reason, current status, temporary allowance, and next action.

## Status Values

- `migration-held`: temporarily allowed in Cellerator until a named migration
  finishes.
- `validation-only`: allowed as a benchmark, comparison, or test aid, but not
  a product/runtime API.
- `review`: needs a decision before new work expands it.
- `legacy-prototype`: kept for compatibility or comparison while a sparse
  biological runtime path replaces it.
- `local-only`: local operations or documentation glue, not packaged runtime
  functionality.

## Entries

| Surface | Current location | Why out of scope | Destination owner | Status | Allowed temporary reason | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| Scanpy/H5AD preprocessing reference utilities | `bench/preprocess_scanpy_reference.py`, preprocessing comparison benchmarks | Scanpy/AnnData comparison is validation/interchange work, not Cellerator execution runtime. | Cellerator preprocessing validation or docs-local validation | `validation-only` | Useful for equivalence checks while preprocessing ownership settles. | Keep out of public runtime APIs; migrate durable validation into Cellerator preprocessing tests or docs-local workflows. |
| Legacy Torch-first dense model prototype | `src/models/dense_reduce/`, related dense-reduce tests | Torch-first dense reduction is not the target Cellerator model style unless it proves a sparse biological operator boundary. | Cellerator examples/prototypes, or rewrite under sparse-native model path | `legacy-prototype` | Provides compile coverage and comparison against newer native sparse paths. | Either rewrite around sparse CellShard batches and Cellerator ops, or move to a non-core prototype/examples area. |
| Legacy Torch custom-op backend | `src/compute/model_ops/`, legacy entries in `custom_torch_ops.md` | General Torch custom-op prototypes should not define the core runtime when native sparse operators are the Cellerator ownership point. | Sparse operator layer in `src/compute/sparse/ops/` or prototype area | `legacy-prototype` | Some model tests still cover old libtorch integration points. | Keep only minimal Torch extension boundaries; move reusable sparse math into framework-independent compute/sparse/ops. |
| Neighbor-caller sibling package | former `CellShardNeighbors/` workspace submodule | Forward-neighbor caller policy has been reclassified as Cellerator-owned compute orchestration over CellShard-backed matrices. | `src/compute/neighbors/forward_neighbors/` | `migration-held` | Kept here only to record the 2026-05-04 ownership reversal during downstream cleanup. | Remove remaining CellStack references to the sibling package and route callers to `Cellerator/compute/neighbors/forward_neighbors.hh`. |
| Local pipeline operations | `docs/pipeline/`, `docs/local_pipeline_ops.qmd` | Local orchestration glue is not product/runtime scope. | Docs-local only, or a separate tool if it becomes reusable | `local-only` | Useful for local datasets and manual pipeline operation. | Do not add CMake/install dependencies on this tree; migrate reusable runtime behavior into the correct owner in a separate pass. |
| Future compiled Cellerator model format | any proposed `.cellerator` format work | The compiled model format is explicitly future work and should not drive current sparse operator interfaces. | Future dedicated model-format workstream | `review` | Discussion is allowed for future planning only. | Do not add file-format APIs or serialization commitments until sparse ops and training primitives are stable. |

## Maintenance Rules

- Add an entry before expanding an out-of-scope surface.
- Close or remove an entry only after the code has moved, been deleted, or been
  explicitly reclassified as in scope in `scope.md`.
- Prefer migration to CellShard for data handling and to Cellerator for
  preprocessing policy and numerical math over CellShard matrices.
- For scope-affecting changes, remind the user that this file is advisory and
  should be reviewed before large moves.
