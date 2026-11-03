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

## Native C++ NN first-order adapters

- Purpose/owner: optional native NN operation adapters for ordinary Torch training; Cellerator NN owns adapters and existing numerical owners retain kernels.
- Status: CPU and admitted V100 correctness validated (11 CUDA groups); separate memory qualification retained in `planning/native-nn-foundation-2026-10-31/derivatives/REPORT.md`.
- C++ boundary: `Cellerator/nn/ops.hh` namespace `cellerator::nn::ops`: CPU `product2_cpu`, CUDA `grouped_linear_cuda` and `csr_scaled_dot_product_attention_cuda`. No new Python binding required.
- Backend: existing native Product2 CPU, cuBLAS grouped linear, staged CUDA CSR attention. No new Tensor Core kernel or fused-small attention route.
- Inputs/outputs: Product2 1D FP32 x/coefficient with int64 gathered indices; grouped linear [R,I]/[E,O,I] -> [R,O] with host row offsets; attention [Q,H,D]/[N,H,D]/[N,H,V] -> [Q,H,V] with explicit CUDA CSR, scale and zero-only dropout.
- Precision/layout: matching CUDA FP16/FP32, FP32 internal accumulation; adapter prepares contiguous storage and preserves original saved tensors. Empty support is zero, duplicate CSR edges remain distinct.
- Autograd/lifetime: first-order VJP only, original version checks, per-invocation native executor/workspace, current guarded stream and allocator stream records. Ordinary Torch parameter ownership and optimizer updates.
- Allocator boundary: Torch `native`/`cudaMallocAsync` backends; foreign from_blob/DLPack owners and native executor handles retained through completion events, outside CUDA callbacks. Arbitrary globally installed pluggable allocators fail clearly. Trusted prepared CSR route reuses validated immutable structure without content reads or snapshots.
- Distributed implications: local operations only, no communication or model ownership policy.
- Code: `src/nn/ops_cpu.cc`, `src/nn/ops_cuda.cc`; tests in `tests/nn/native/ops*.cc`.

## `factorized_interaction_cuda`

- Purpose: expose the native gated joint-interaction law to ordinary first-order
  ATen autograd while retaining the native packet/direct CUDA implementation.
- Owner: Cellerator's optional C++ ATen adapter over the native CUDA component.
- Status: C++ ATen and optional Python prepared-plan adapters are implemented
  and compile-checked. An admitted V100 run passed the prepared adapter's FP32
  gradients, parameter updates, cross-stream reuse and mutation/lifetime checks;
  this does not qualify every adapter dtype combination. Native mixed-storage
  and sanitizer coverage is recorded separately in
  `planning/interaction-layout-proposal-2026-11-02/VALIDATION.md`.
  Performance qualification is paused with no accepted timing results.
- Python boundary: `_nn_native.PreparedFactorizedInteraction` accepts records
  with ordered inputs and context terms plus explicit extents. Its first CUDA
  use uploads and caches one immutable structural owner per device; subsequent
  calls reuse that owner. The Python function only selects output dtype/route
  and passes tensors to native autograd.
- C++ boundary: `factorized_interaction_cuda_prepared` accepts a shared native
  `FactorizedInteractionDeviceLayout` and five CUDA tensors. The host-layout
  `factorized_interaction_cuda` overload remains a cold convenience path that
  validates and uploads structure for one call.
- Backend: regular custom CUDA for the irregular interaction and its VJP;
  this law is not a dense matrix multiplication or Tensor Core GEMM. The native
  target remains independently usable without Torch.
- Inputs/outputs: `x[input,channel,cell]`, `z[context,cell]`,
  `alpha[parameter_row,channel]`, `bias[parameter]`, and `gamma[parameter]`;
  output is `y[target,channel,cell]` with zero base and additive overlapping
  contributions. Ordered/repeated roles, context terms, and tied parameters
  retain their declared multiplicity.
- Precision/layout: contiguous CUDA FP16 or FP32 activations, matching context
  dtype, and independently selected matching parameter dtype; explicit FP16 or
  FP32 output. Native arithmetic and VJP destinations are FP32. The adapter
  returns each gradient in its original tensor dtype.
- Autograd/lifetime: first-order VJP only; original tensors are saved for
  version checks; per-forward gate tape and scratch remain live through queued
  use; the guarded current stream is used and borrowed tensors are recorded or
  retained through a completion event. Higher-order differentiation is not
  supported.
- Distributed implications: local execution only; no communication, routing,
  or optimizer ownership is added.
- Code: `include/Cellerator/nn/ops.hh`, `src/nn/ops_cuda.cc`, and
  `bindings/nn/module.cc`, backed by
  `include/Cellerator/modeling/cuda/factorized_interaction.hh`.
