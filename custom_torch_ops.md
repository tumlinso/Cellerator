# Custom Torch operations

This document records numerical extension contracts. It is not a runtime
registry or a required model hierarchy. Ordinary Torch compositions need no
registration here. Existing operation owners and their evidence remain
authoritative.

## `grouped_linear`

- Purpose: execute compact expert-major or module-major linear transformations
  inside ordinary differentiable biological models.
- Owner: Cellerator's optional Torch adapter over the native grouped-linear
  operation.
- Status: validated for bounded first-order CPU/reference and native V100 CUDA
  composition; no speedup or whole-model biological claim.
- Python boundary: `grouped_linear(inputs, weight, row_offsets,
  implementation="auto")`, with an optional `GroupedLinear` module.
- C++ boundary: existing `_modeling_cuda` device views, borrowed streams, and
  `GroupedLinear` forward/backward; no new numerical kernel is required.
- Backend: Torch reference composition on CPU; existing cuBLAS-backed native
  CUDA execution, including SM70. Explicit reference execution is also available.
- Inputs: features `[N, I]`, one Torch-owned weight `[E, O, I]`, and immutable
  host row offsets of length `E + 1`. Empty expert segments and `N = 0` are valid.
- Outputs: expert-major features `[N, O]` with the same row ordering.
- Precision/layout: matching FP16 or FP32 operands; native FP32 accumulation;
  contiguous borrowed views, with explicit differentiable conversion for
  noncontiguous inputs. No implicit dtype conversion.
- Autograd: first-order input and weight VJPs; integer structure is not
  differentiable. Torch retains parameter ownership and saved-tensor version
  checks. Native higher-order differentiation is unsupported.
- Lifetime/streams: Torch allocations and the current stream; one native
  executor per forward graph, retained through backward with borrowed tensors.
- Distributed implications: this is local numerical execution, not a transport
  or parameter-residency service. Routing and reverse redistribution are separate.
- Code: `python/cellerator/torch/grouped_linear.py`, existing native
  `include/Cellerator/modeling/cuda/grouped_linear.hh` and
  `src/modeling/cuda/grouped_linear.cu`.
- Validation: seven CPU/reference tests; admitted V100 numerical and memcheck
  runs each passed all 13 tests with zero skips. Memcheck reported zero errors,
  and both jobs verified cleanup. See
  [the source-bound report](planning/nn-foundation-2026-10-31/grouped_linear/REPORT.md).
  Native higher-order differentiation, automatic mixed-precision registration,
  compiler integration, and performance remain outside this qualification.
