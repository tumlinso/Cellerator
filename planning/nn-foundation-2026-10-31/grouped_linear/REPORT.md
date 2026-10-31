# Grouped-linear NN integration, 2026-10-31

The new Torch adapter composes the existing native CUDA grouped-linear
operation with ordinary Torch parameters, activations, optimizers, and state
dictionaries. No C++ binding or CUDA numerical implementation was changed.

## Result and evidence

- Standalone CPU discovery: seven reference/contract tests passed; six CUDA
  tests were intentionally skipped without admitted CUDA execution.
- Admitted V100 numerical run: all 13 tests passed, zero failures, errors, or
  skips. Job `9bf9e2bc-a273-4491-9802-0bfe317e6f05`, attempt
  `ddb2a703-728c-4678-94ac-5ed192fa6771`.
- Admitted Compute Sanitizer memcheck: all 13 tests passed with
  `ERROR SUMMARY: 0 errors`. Job `8442581b-0a52-4373-b812-61d8e79c225f`,
  attempt `a49f689c-8359-41a8-bd1d-02293c6f4570`.
- Both executed jobs report `cleanup_verified=true`. They exclusively used
  `GPU-d745da9c-7649-8334-41f1-483afa6f3206`, Tesla V100-SXM2-16GB, SM70.

The first requested UUID was not feasible for admission. Job
`67d5df76-8e01-482f-be25-791d0aab1187` had no attempt or lease and was cancelled
with verified cleanup before rerouting. No unmanaged CUDA fallback was used.
The existing Cellerator scheduler client was selected explicitly; no credentials
or service configuration were changed.

[`evidence/cuda-run-2.json`](evidence/cuda-run-2.json) and
[`evidence/cuda-memcheck-run-1.json`](evidence/cuda-memcheck-run-1.json) bind Python,
Torch, GPU, extension SHA256, and relevant source SHA256. Adjacent job payloads,
submissions, status records, and logs retain admission and cleanup evidence.
[`evidence/guidance-sha256.json`](evidence/guidance-sha256.json) identifies the
consumed CUDA and KDA guidance; KDA revision was
`277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc`.

## Implementation

- `python/cellerator/torch/grouped_linear.py` provides `grouped_linear` and
  `GroupedLinear` without creating a second parameter owner.
- Host offsets describe expert-major segments. Torch reference execution and
  native CUDA execution implement the same selected expert matrix maps.
- Native forward saves operands for version checks and retains its scoped
  executor. Native backward writes into Torch-allocated input/weight gradient
  buffers. Storage is borrowed on the current Torch stream and recorded with
  its allocator. Noncontiguous inputs are explicitly normalized before entering
  the native autograd operation.
- CUDA supports FP16/FP32 operands and outputs with FP32 internal accumulation.
  Ordinary differentiable casts connect FP32 master weights to FP16 compute.
  Native differentiation is first-order only. Reference higher-order gradients
  pass FP64 gradcheck and gradgradcheck.

## Coverage

Independent row-wise dot-product references check ragged/empty experts, input
and weight VJPs, zero total rows, noncontiguous operands, and zero gradients for
unused experts. Tests exercise shared weights across multiple calls, ordinary
SGD/AdamW, state dictionaries, non-default CUDA streams, FP32 masters with FP16
compute, saved-parameter mutation rejection, higher-order rejection, and three
composed training steps with ordinary learned layers before and after the native
operation.

FP32 comparisons use `rtol=3e-5, atol=3e-5`; represented-input FP16 comparisons
use `rtol=6e-3, atol=4e-3` for these bounded small fixtures. They are not a
whole-library precision guarantee.

## Reproduction

CPU:

```sh
PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests/framework -p test_nn_grouped_linear.py -v
```

The new minimal build harness includes the existing framework CUDA CMake owner.
The qualification build was `/tmp/cellerator-nn-grouped-linear-sm70`, targeting
only architecture 70 with CUDA 12.9.86, GCC 12.4, Python 3.13.11, and pybind11
3.0.3. Torch was `2.7.0+cu126`. Submit the retained job payload through the
authenticated Device Scheduler after rebuilding; the runner appends that build
to the source package search path and does not install or replace the package.

## Limits

These are correctness and memory-access checks, not performance measurements,
biological validation, or a distributed-training qualification. No new kernel,
profiling campaign, or parameter-residency system was introduced. One native
executor/cuBLAS handle is prepared per autograd invocation; preparation costs
remain part of the operation's complete cost. CUDA higher-order differentiation,
automatic mixed-precision registration, compiler integration, and additional
devices/precisions are not claimed.
