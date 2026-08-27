# Torch Bindings

This directory is an explicit bridge from `Cellerator` / `CellShard` data
structures into Torch-facing code.

It exists for one reason:

- let downstream ML code treat Cellerator as a library without pushing Torch
  abstractions down into the storage, staging, and runtime core

It does **not** exist to redefine the core execution model of the project.

## What This Layer Is

This layer is for:

- exporting already-materialized host-side sparse data to `torch::Tensor`
- giving libtorch-based models a narrow and predictable integration boundary
- making interop possible without rewriting CellShard around Torch

## What This Layer Is Not

This layer is not:

- a replacement for CellShard storage
- a replacement for sharded staging
- a replacement for owner-local CUDA execution
- a replacement for sparse preprocess kernels
- something you should route hot-path runtime code through

## Why The Warnings Are So Strong

The bridge helpers here are intentionally expensive and intentionally obvious.

They do real work:

- allocate fresh Torch-owned tensors
- widen CellShard sparse indices from 32-bit to Torch 64-bit indices
- copy sparse metadata
- copy value payloads
- stitch parts together when exporting a sharded matrix

That means the helpers are appropriate at explicit interop boundaries, for
example:

- exporting a dataset once for a Torch-native prototype
- handing a sampled matrix to a model library outside the Cellerator runtime
- building a thin library API for downstream research code

That also means they are **not** appropriate in:

- per-step training loops
- repeated minibatch export inside hot code
- preprocessing internals
- staging internals
- multi-GPU owner-local execution code

## Practical Rule

If a call site would be performance-sensitive in Nsight Systems, it probably
should not use this directory.

Good boundary:

- `CellShard / Cellerator` does storage, staging, sparse runtime, and custom
  CUDA work
- `components/CelleraTorch/` exposes an explicit conversion point
- Torch-native models start on the other side of that conversion

Bad boundary:

- convert to Torch first
- then do core sparse runtime work in Torch because the tensor exists

That second pattern is exactly what this directory is trying to prevent.

## Preferred native adapter

The CE-LIVE adapter is the preferred repeated-execution boundary:

```text
Cellerator-owned storage and parameters
        -> lifetime-bound Torch views
        -> current Torch CUDA stream
        -> prepared Cellerator executable/training program
        -> explicit output order and readiness metadata
```

`CelleraTorch::native_adapter` publishes the three independently tested target
surfaces `CelleraTorch::native_views`, `CelleraTorch::program_ops`, and
`CelleraTorch::autograd_ops`. These are C++/libtorch operation wrappers; they do
not introduce a second planner, runtime session, parameter store, or numerical
kernel implementation.

The forward wrapper binds caller-owned Torch tensors to a prepared Cellerator
program and uses the current Torch CUDA stream. The autograd wrapper preserves
the current native N=16 training envelope, returns the dense-input gradient,
and leaves relation-value and bias updates under native Cellerator ownership.
Both surfaces preserve native structure epoch, value generation, output-order,
and readiness contracts.

No adapter target calls the copied CSR exporter from `bindings.hh`. That API is
retained by `CelleraTorch::bindings` for compatibility and debugging, where its
allocation, index widening, and copying costs are intentional and visible.

Shared build integration is enabled only by
`CELLERATOR_ENABLE_TORCH_MODELS=ON`. The native Cellerator default remains
Torch-free. The combined `celleraTorchNativeAdapterTests` build target compiles
all three adapter lanes; run its three executables to validate the integrated
surface:

```bash
./build-celleratorch/celleraTorchNativeViewsTest
./build-celleratorch/celleraTorchProgramOpsTest
./build-celleratorch/celleraTorchAutogradOpsTest
```
