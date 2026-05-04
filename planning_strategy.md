# Cellerator Planning Strategy

Last updated: 2026-04-28

This document is a pickup guide for turning Cellerator toward a sparse
biological ML runtime. It assumes the scope boundary in `scope.md` and the
advisory migration queue in `out_of_scope_inventory.md`.

## Planning Principle

Plan Cellerator around operator contracts before model families, file formats,
or package ergonomics.

Cellerator should become useful because it owns sparse biological training
primitives that Torch does not express efficiently over CellShard-scale data:
Blocked-ELL/Sliced-ELL execution, sparse operator, pathway/module reductions,
graph/neighborhood operations, quantized sparse reconstruction, and deliberate
Torch interop.

The compiled `.cellerator` model format stays out of scope until the operator
and training contracts are stable.

## Phase 1: Define The Runtime Contracts

Start here before implementing new models.

1. Define the sparse batch ABI.
   - Inputs: rows, features, assay id, part/shard id, layout, device, stream.
   - Layouts: Blocked-ELL primary, Sliced-ELL secondary, CSR fallback.
   - Residency: spell which buffers are device-resident and which metadata is
     host-side.
   - Metadata: feature ids, pathway ids, graph ids, and assay ids must be
     attachable without entering hot kernels accidentally.

2. Define the operator registry.
   - Use a table with: op name, biological purpose, layouts, forward contract,
     backward contract, saved state, Torch exposure, and owner target.
   - Initial candidate ops: sparse projection/SpMM, row reductions, feature
     reductions, pathway/module reduction, graph aggregation, sparse
     reconstruction loss, quantized sparse affine/reconstruction, and sparse
     gather/scatter.

3. Define the differentiation boundary.
   - Decide per op whether Cellerator owns forward only, forward plus backward,
     or forward/backward/optimizer state.
   - Be explicit about which gradients exist: values, weights, gene parameters,
     latent values, graph weights, pathway maps, or no metadata gradients.
   - Prefer framework-independent C++/CUDA contracts first; Torch custom ops
     should wrap stable contracts rather than define them.

Deliverable: add `operator_scope.md` or `docs/operator_contracts.qmd` with the
operator table and ABI rules.

## Phase 2: Separate Product Work From Migration Work

Do not let cleanup masquerade as feature design.

1. Use `out_of_scope_inventory.md` as the advisory migration queue.
   - Preprocessing implementation moves to CellShardPreprocess.
   - Ingest, layout construction, and pack/runtime publication move to
     CellShard.
   - Legacy Torch-first dense prototypes are either rewritten around sparse
     batches or moved out of the core runtime.

2. Keep migration targets measurable.
   - A moved surface should retain or improve its current compile/runtime test.
   - Cellerator should keep only wrapper, interop, or benchmark glue when the
     owner is CellShard or CellShardPreprocess.

3. Do not add new features to migration-held code unless the same change is
   part of moving it to the right owner.

Deliverable: add one workstream per migration group only when implementation
starts; do not overload the main operator-design plan with cleanup details.

## Phase 3: Build The First Useful Sparse Training Stack

Pick one narrow vertical path and make it real end to end.

Recommended first vertical:

1. Sparse batch view from CellShard execution data.
2. Sparse projection/SpMM plus row/pathway reduction.
3. Native backward for the projection/reduction path.
4. One biological primitive model that uses the ops directly.
5. Optional Torch wrapper around the stable op/model boundary.

Good first model candidates:

- pathway-aware sparse reduction layer
- developmental-time sparse head over CellShard batches
- graph/neighborhood smoother with sparse latent state
- quantized sparse reconstruction path

Avoid starting with a broad model zoo. The first model should prove that the
operator ABI, differentiation boundary, and CellShard batch contract are correct.

## Phase 4: Define Torch Interop After The Native Contract

Torch is the integration layer, not the execution owner.

Plan interop in this order:

1. Explicit export for deliberate conversion to Torch tensors.
2. Torch custom ops for stable Cellerator operators.
3. PyTorch modules that call Cellerator ops without hiding layout conversion.
4. DLPack or Python bindings only where they preserve ownership boundaries.

Rules:

- No hidden per-step conversion through Torch for hot sparse training.
- No Torch-only implementation of a Cellerator-owned sparse operator unless it
  is explicitly a prototype.
- Record new custom-op boundaries in `custom_torch_ops.md` before
  implementation.

## Suggested Pickup Order

1. Create the operator contract document.
2. Inventory existing compute/model targets against that contract.
3. Pick one vertical sparse training path.
4. Define tests for its forward, backward, and layout behavior.
5. Implement the smallest native path.
6. Add Torch exposure only after the native path is stable.
7. Revisit `out_of_scope_inventory.md` and split cleanup workstreams.

## Decision Checklist

Before accepting a new Cellerator surface, answer:

- Does it operate on CellShard-scale sparse biological data?
- Does it need Blocked-ELL, Sliced-ELL, CSR fallback, cuSPARSE, or custom CUDA?
- Is it training-capable, or does it support a training-capable primitive?
- Is Torch insufficient because of layout, differentiation, residency, or distributed
  sparse execution?
- Does the surface belong in CellShard or CellShardPreprocess instead?
- Is it future `.cellerator` model-format work? If yes, defer it.

If the answer is unclear, add a `review` entry to
`out_of_scope_inventory.md` instead of expanding Cellerator ownership.

## Recommended Next Artifact

Create `operator_scope.md` with this structure:

| Op | Biological purpose | Layouts | Forward | Backward | Saved state | Torch exposure | Owner |
| --- | --- | --- | --- | --- | --- | --- | --- |

Seed it with sparse projection, row reduction, pathway/module reduction, graph
aggregation, sparse reconstruction loss, quantized sparse affine, and
gather/scatter. That document should become the planning backbone for the next
implementation pass.
