# Research precedent and source index

## Live Cellerator source anchors

The plan works backward from current Cellerator machinery, including:

- `include/Cellerator/compute/operation/operation_core_v2/`
- `include/Cellerator/compute/operation/relation_algebra_v2/`
- `include/Cellerator/compute/operation/candidate_catalog_v3/`
- `include/Cellerator/compute/decomposition/`
- `include/Cellerator/geometry/compiler/v2/`
- `include/Cellerator/geometry/optimizer/`
- `include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh`
- `include/Cellerator/geometry/persistence/execution_image_v2.hh`
- `include/Cellerator/execution/joint_compiler/`
- `include/Cellerator/execution/object_binding/`
- `include/Cellerator/execution/lowering_resumption/`
- `include/Cellerator/execution/program/`
- `include/Cellerator/execution/training_program_v2/`
- `include/Cellerator/execution/projection_value_plane/`
- `include/Cellerator/planner/`
- `include/Cellerator/profiling/joint_compiler/`
- `include/Cellerator/runtime/`
- `planning/jbc-preledger-v1/`

The exact discovered file/worktree inventory is machine-readable under `inventories/` and `evidence/`.

## Compiler precedents consulted

Primary documentation was used for engineering precedent, not copied as language identity:

- LLVM Language Reference and bitcode/object/LTO documentation;
- MLIR language reference, dialect, bytecode, pass-management, and dialect-conversion documentation;
- Clang LibTooling, frontend action, preprocessing/pragma, AST, plugin, and clangd documentation;
- GCC GENERIC/GIMPLE/plugin/LTO documentation where representation and compiler-driver behavior are relevant;
- NVIDIA NVCC compilation phases, CUDA separate compilation/device linking, PTX ISA, and ptxas documentation;
- CMake compiler, package export, install, depfile, and custom-language integration documentation;
- Language Server Protocol documentation;
- current C++ reflection and compile-time-programming proposals as design precedent, without depending on unstandardized reflection.

## What was borrowed

- explicit staged driver action graphs;
- upstream C++ semantic reuse behind adapters;
- public textual IR and exact IR-text round trip;
- dialect/extension preservation;
- pass pipelines and analysis invalidation;
- companion object IR and LTO summaries;
- typed inline-native contracts;
- language-server incremental snapshots.

## What was deliberately not borrowed

- LLVM/MLIR textual aesthetics as Cellerator syntax;
- a generic untyped graph that adds biology after lowering;
- a permanent Clang fork;
- a safe-only plugin surface;
- opaque planner decisions;
- target details in Semantic IR;
- compulsory bespoke executable containers;
- runtime metadata overhead for compile-time provenance.

## Current specification inputs

See `evidence/language_document_inventory.json` for exact paths and hashes.
