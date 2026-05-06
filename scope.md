# Cellerator Scope Boundary

Last updated: 2026-05-04

## Purpose

Cellerator is a sparse biological ML runtime and operator library for training
and inference on CellShard-scale omics data. Its dependency-light
CelleratorCore layer owns sparse layout primitives, GPU operators, sparse
differentiation and training primitives, biologically structured model
components, distributed sparse execution, and explicit Torch interop boundaries
that Torch does not natively provide well for sparse biological structure.

Cellerator should extend Torch where sparse omics execution needs new
capability. It should not replace Torch as the general ML framework.

## Ownership Boundary

- CelleratorCore owns sparse layout primitives and is the migration target for
  generic sparse math: Blocked-ELL, Sliced-ELL, quantized Blocked-ELL, layout
  helpers, device views, sparse GPU operators, reductions, transforms, sparse
  training primitives, forward-neighbor index/query policy over CellShard-backed
  matrices, and quantized sparse kernels. Some compatibility runtime wrappers
  may remain in CellShard during migration.
- Higher-level Cellerator owns biological model components, trajectory/model
  math, and explicit Torch/libtorch extension or export boundaries.
- CellShard owns data handling over CelleratorCore payloads: `.csh5`,
  `.cshard`, CSPACK generations, source ingest, retrieval, sharded runtime
  staging, compaction/finalization, and pack publication.
- Cellerator preprocessing owns biology-facing preprocessing policy and APIs:
  QC metric definitions, normalization/log1p workflow semantics, feature/cell
  keep-mask semantics, raw-count validation, adapter staging, and preprocessing
  workbench/runtime orchestration. Reusable numerical kernels over CellShard
  matrices live in `src/compute/preprocess/` and are called through explicit
  layout-aware boundaries.
- AnnData/H5AD remains an ecosystem interchange and workflow format.
  Cellerator may import from or export to it through CellShard-facing paths,
  but AnnData is not the execution format.
- Torch remains the general tensor ML framework. Cellerator may expose custom
  ops, DLPack/tensor exports, libtorch modules, and Python bindings, but hot
  sparse runtime work should remain layout-explicit and Cellerator-owned.

## In Scope

- Sparse operators for Blocked-ELL, Sliced-ELL, CSR fallback, cuSPARSE-backed
  paths, sparse-dense projections, reductions, gather/scatter, row/feature
  selection, exact-search/scoring math, pathway/module reductions, and graph-aware
  sparse transforms.
- Training-capable sparse operator where Torch sparse support is missing,
  layout-mismatched, or too costly for CellShard-scale data.
- Biological model primitives that make sparsity part of the model: pathway
  layers, gene/module reductions, sparse latent models, developmental-time and
  trajectory components, graph/neighborhood losses, and quantized sparse
  reconstruction paths.
- Distributed sparse execution when shard layout, data sparsity, or model
  structure makes multi-GPU execution materially useful.
- Explicit interop with Torch and major ML packages at deliberate boundaries,
  not hidden conversion inside hot loops.

## Out Of Scope For Now

- A compiled `.cellerator` model format. This is future work and must not drive
  near-term interfaces.
- Replacing Torch, PyTorch, JAX, Scanpy, AnnData, or scvi-tools.
- Generic sparse matrix library work that is not tied to sparse biological ML
  training or CellShard execution layouts.
- Source ingest, durable storage, pack generation, and runtime publication
  outside the CellShard owner surface.
- Graph construction workflows that are not part of Cellerator-owned model,
  trajectory, or forward-neighbor search execution.
- High-level workflow or notebook ergonomics that hide layout, residency,
  transfer, or launch costs in performance-sensitive paths.

## Agent Rules

- Before adding a public API, CMake target, model module, preprocessing path,
  ingest/runtime path, or Torch bridge, check this file and
  `out_of_scope_inventory.md`.
- For roadmap work that changes what Cellerator should become next, use
  `planning_strategy.md` as the pickup guide.
- If a new surface is outside this scope but temporarily necessary, add or
  update an inventory entry instead of normalizing it as Cellerator-owned.
- If a surface is data handling, prefer migration to CellShard. If a surface is
  preprocessing policy or numerical preprocessing math over CellShard matrices,
  keep or migrate it into Cellerator.
- If work touches scope drift, remind the user that
  `out_of_scope_inventory.md` is the advisory migration queue.
