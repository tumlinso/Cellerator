# CelleraTorch

CelleraTorch is the Torch-facing Cellerator sister component. Its purpose is to
isolate Torch and libtorch dependencies from Cellerator sensu stricto, which
remains the base sparse biological math library.

## Intended Ownership

CelleraTorch should own Torch-facing integration:

- libtorch and PyTorch build/linking glue
- Torch tensor export and import boundaries
- DLPack-style interop, if needed
- custom Torch op wrappers around native Cellerator primitives
- Torch-backed model adapters
- Python package extras that depend on Torch

It should consume Cellerator for math and CellShard for storage/staging. It
should not become a second home for sparse kernels, quantized layout mechanics,
preprocessing policy, or CellShard data handling.

## Current Surface

Current CelleraTorch surfaces include:

- `CelleraTorch::bindings`
- `CelleraTorch::model_ops`
- `CelleraTorch::quantize`
- `CelleraTorch::dense_reduce`
- `CelleraTorch::torch`

Reusable math discovered during migration should stay in native Cellerator
under the appropriate compute/runtime layer, with CelleraTorch holding only the
Torch adapter around it.

## Parameter Exposure Direction

CelleraTorch should become the efficient Torch view over learned parameters that
native Cellerator owns. The native side should expose Torch-free non-owning
descriptors for parameter name, scalar type, host/device residency, device
ordinal, raw pointer, shape, stride, writability, and role. The current native
hook for that contract is `include/Cellerator/core/parameters.hh`.

Future CelleraTorch adapters should use those descriptors to create Torch tensor
views over native Cellerator storage without copying where possible. That means
allocator work in Cellerator needs to preserve pointer stability, explicit
contiguity decisions, and correct stride metadata. CelleraTorch can interpret
and wrap those allocations for Torch, but it should not become the owner of the
canonical learned-parameter buffers.

## Build Status

This component is wired through the existing `CELLERATOR_ENABLE_TORCH_MODELS`
CMake option. Disable that option for a native Cellerator build without
Torch package discovery.
