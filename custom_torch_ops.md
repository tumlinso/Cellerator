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

## Native C++ NN first-order adapters (planned)

- Purpose/owner: optional native NN operation adapters for ordinary Torch training; Cellerator NN owns adapters and existing numerical owners retain kernels.
- Status: implementation in progress; qualification pending in `planning/native-nn-foundation-2026-10-31/derivatives/`.
- C++ boundary: `Cellerator/nn/ops.hh` namespace `cellerator::nn::ops`: CPU `product2_cpu`, CUDA `grouped_linear_cuda` and `csr_scaled_dot_product_attention_cuda`. No new Python binding required.
- Backend: existing native Product2 CPU, cuBLAS grouped linear, staged CUDA CSR attention. No new Tensor Core kernel or fused-small attention route.
- Inputs/outputs: Product2 1D FP32 x/coefficient with int64 gathered indices; grouped linear [R,I]/[E,O,I] -> [R,O] with host row offsets; attention [Q,H,D]/[N,H,D]/[N,H,V] -> [Q,H,V] with explicit CUDA CSR, scale and zero-only dropout.
- Precision/layout: matching CUDA FP16/FP32, FP32 internal accumulation; adapter prepares contiguous storage and preserves original saved tensors. Empty support is zero, duplicate CSR edges remain distinct.
- Autograd/lifetime: first-order VJP only, original version checks, per-invocation native executor/workspace, current guarded stream and allocator stream records. Ordinary Torch parameter ownership and optimizer updates.
- Distributed implications: local operations only, no communication or model ownership policy.
- Code: `src/nn/ops_cpu.cc`, `src/nn/ops_cuda.cc`; tests in `tests/nn/native/ops*.cc`.
