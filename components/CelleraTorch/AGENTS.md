# CelleraTorch Agent Guide

## Scope

CelleraTorch is the staging component for the future Torch-facing Cellerator
sister project. It owns Torch and libtorch integration boundaries that should
not remain in Cellerator sensu stricto once the base math library split is
complete.

Keep reusable sparse math, kernels, layouts, reductions, quantized primitives,
runtime scratch, and CUDA library mechanics in native Cellerator. CelleraTorch
should consume those primitives through stable Cellerator APIs instead of
copying or redefining them.

CelleraTorch is allowed to own:

- Torch/libtorch linking and build glue.
- Torch tensor export/import helpers.
- DLPack or related Torch interop boundaries.
- Torch custom-op wrappers around Cellerator primitives.
- Torch-backed model adapters and packaging.
- Python package extras that require Torch.

CelleraTorch should not own:

- CellShard storage, ingest, pack publication, or runtime staging.
- Cellerator base sparse math or quantized kernels.
- Preprocessing policy or workflow semantics.
- Neighbor or trajectory caller policy unless the code is specifically a
  Torch adapter around another owner surface.

## Current Status

This directory owns the current Torch-facing build, headers, wrappers, and tests
that were split out of native Cellerator. Preserve the native Cellerator API
boundary: if a change discovers reusable sparse math, add it to native
Cellerator first and keep only the Torch adapter in this component.
