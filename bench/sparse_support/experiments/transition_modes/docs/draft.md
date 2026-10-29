# Transition modes implementation draft

## Task contract

- Task: CE-COHERE transition parameter semantics, shared and per-mechanism modes.
- Objective: add an explicit mode-aware native/Python transition operation while
  preserving the current v1 API, stacked flat parameter owner, stock Torch
  optimizer state, and legacy checkpoint behavior.
- Target: existing Cellerator CUDA/Torch extension, initially qualified for
  Volta `sm_70`; compile-only work is allowed now. GPU execution, probes,
  sanitizer runs, and timing require root admission.
- Workload: latent state `[cells, mechanisms, width]`, input `[cells, input_width]`,
  unique sorted affected mechanism IDs per cell, and flat FP32 master parameters
  stored as stacked `A[M,W,W]`, `B[M,W,I]`, and `bias[M,W]` blocks.
- Semantics: in `shared`, all affected mechanisms use block zero of A/B/bias.
  In `per_mechanism`, mechanism `m` uses block `m`. Unaffected state is the
  identity path. The independent detached route index continues to inspect all
  B blocks in both modes; this is a separate legacy policy.
- Numerics: input and parameters are stored as FP32 Torch/native values and
  converted to FP16 operands for the current native transition; multiply is
  FP16, accumulation/output and gradients are FP32, and optimizer master values
  remain FP32. The independent C++/Python references use FP32 arithmetic.
- Correctness: forward, state VJP, input VJP, A/B/bias VJPs; multiple cells and
  mechanisms; untouched states; invalid mode/shape/dtype/device; explicit mode
  mismatch rejection before checkpoint mutation; continuation of old and tagged
  checkpoints including named stock-optimizer state and canonical leaf identity.
- Independent oracle: a test-side direct scalar implementation of the stated
  tanh equation and derivatives, separate from the native reference and CUDA
  kernels. Native CPU references are also tested against that independent oracle.
- Baseline: `sparse_latent_transition` v1 and `latent_transition_*_v1` preserve
  block-zero shared arithmetic. Detached routing in `SparseSupportModule` uses
  the full stacked `transition_B`. Existing sparse checkpoint payloads have
  flat values and block metadata, with no transition mode tag.
- Source identity at draft: repository checkpoint `1d1dfda8c5b5ace00415c9bd0847f975f6b655b4`;
  public header SHA-256 `308c1c7713a1346db11359f633a676c533e88221785ee83f4bc0681317231bf1`;
  CUDA implementation SHA-256 `7aee96dabf361d7ed68346edc18489daf78b8a492ed1536b76e574adff791f26`;
  Torch binding SHA-256 `dc5c15c2f315b1eec89c28d86b5893166c575fca7a2ccb4f376a6bbd8c46d4cd`.
- Performance target: none. This change adds a semantic capability; no speedup
  is claimed or measured.
- Allowed implementation: additive native v2 header/reference/CUDA entry points,
  an additive Torch autograd binding, Python transition views and mode metadata.
  Keep v1 symbols/body behavior, one canonical flat owner, and existing stock
  optimizer/checkpoint pathways.
- Validation: CPU reference tests and host build; build the CUDA/Torch extension
  if the configured compiler is available. GPU tests and Compute Sanitizer are
  required later under root admission and are not run in this phase.
- Promotion: pass independent forward/VJP checks and mode/shape validation,
  checkpoint preflight/continuation tests, CPU reference suite, Python source
  tests, and successful extension build. GPU qualification remains explicitly
  pending until admitted.
- Consumed guidance identity: KDA checkout `277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc`;
  CUDA skill SHA-256 `3992c5b0e85409cce8002fcfe2c6caed63df6451daad9bae28ccd64a381d717a`;
  Torch extension guide SHA-256 `3f17b75c30401ec43e551aa213730614237455432699bdf96b9db7780b4cdaff`;
  sparse workloads guide SHA-256 `81672cd93c3d50aa002cda559eabcbddcce873ec2f2824f40168e47f97382df1`.

## Implementation steps

1. Add a v2 parameter mode contract without changing the v1 entry point.
2. Add a host reference that uses a mode-specific parameter block offset and
   retains identity gradients for untouched mechanisms.
3. Add v2 CUDA forward/VJP kernels with compact shared or explicitly stacked
   parameter strides; preserve current FP16 operand and FP32 accumulation
   boundaries.
4. Bind the v2 op with explicit dtype, rank, extent, contiguity, and device
   checks, and return gradients with the exact input parameter shape.
5. Add immutable Python constructor mode, `transition_parameters()` and
   `routing_parameters()` accessors, and route both model consumers through v2.
6. Tag sparse checkpoint state with mode, storage layout, and routing policy;
   interpret untagged payloads as legacy shared/block-zero dynamics with
   all-B routing. Reject explicit mode mismatch during preflight, before owner
   or optimizer state mutation.
7. Add CPU scalar-oracle and Python validation; compile all supported host and
   CUDA/Torch targets. Leave admitted GPU execution and sanitizer commands for
   root.

## Risks and unresolved validation

- The old binding accepts rank-three parameter tensors but v1's native pointer
  indexing consumes compact block-zero A/B/bias. This is the compatibility
  behavior being preserved, not a per-mechanism implementation.
- CUDA gradient accumulation across cells sharing a mechanism must remain
  race-free. The implementation must use the existing reduction/atomic pattern
  or a correct staged reduction and later pass racecheck.
- `B` tails simultaneously own legacy detached route keys. Shared transition
  views must not erase them, and tagged checkpoints must distinguish route
  policy from transition mode.
- GPU kernel correctness, memory/race/synchronization sanitizers, and stream
  behavior are not qualified until root admits the relevant runs.

## Implementation snapshot and validation record

- Implementation sources at the CPU/native and extension build:
  - `include/Cellerator/ops/sparse_support/training.hh` SHA-256
    `5aacc50cf4feb96829c567ab686d3574252a8c197943670e4bff7a17f083e069`
  - `src/ops/sparse_support/training.cc` SHA-256
    `32dd83d15ba387cf0364351a5793e6a1d2f584f5f7b58172f5ad210bae6aa793`
  - `src/ops/sparse_support/training.cu` SHA-256
    `4f43f6c4a1ce2e0e43a2e8775bb955f1a20a6923607a6e21c79795dd218491e8`
  - `bindings/sparse_support/module.cc` SHA-256
    `dd0aaf8236c1909b61b4107e28c6404343310c7d143089f53e08f1866bbc0d49`
- Python integration/test identities:
  - `python/cellerator/torch/execution.py` SHA-256
    `adcaca5c59a239349f4454928f9a9fb441e0f6e20b7031be8cc02af4387d9fbf`
  - `python/cellerator/torch/routing.py` SHA-256
    `dcac5c6e812cbc169e28d6f46461d8a3212bd2e280040f46b1ab233b81837a4e`
  - `python/cellerator/torch/ops.py` SHA-256
    `23b3b0466cfb31c5e9e81af8dfedc7ae751aa88b35d761dbefe2a0f20ccc0609`
  - `tests/sparse_support/numeric_reference.cc` SHA-256
    `e72fa510428550d46b4f9d1e131814693db543ba64d8b4909019fc8f9c5ee190`
  - `tests/sparse_support/test_transition_modes.py` SHA-256
    `6c7143206338d699cd9df7c5d3d50b7df3c1f2397ce3481f90b8500a4e2a3979`
  - `tests/sparse_support/test_cuda_training.py` SHA-256
    `244247f09e659931ab7ebb1f80e48a5779a2d23a5d54953fb7d6c07a25329ca0`
- Configured `/tmp/cellerator-transition-cuda-build` with CUDA ON, Torch ON,
  hardware probe OFF, auto architecture detection OFF, and
  `CMAKE_CUDA_ARCHITECTURES=70`. No GPU was probed or used.
- Successful commands:
  - `cmake --build /tmp/cellerator-transition-cuda-build --target cellerator_python_torch cellerator_sparse_support_numeric_reference -j2`
  - `/tmp/cellerator-transition-cuda-build/tests/sparse_support/cellerator_sparse_support_numeric_reference`
  - `python -m pip install . --no-build-isolation --no-deps --target /tmp/cellerator-transition-install-final --config-settings=cmake.define.CELLERATOR_ENABLE_HARDWARE_PROBE=OFF --config-settings=cmake.define.CELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES=OFF`
  - `PYTHONPATH=/tmp/cellerator-transition-install-final python -m unittest discover -s tests/sparse_support -p 'test_transition_modes.py' -v` (5 passed)
  - `PYTHONPATH=/tmp/cellerator-transition-install-final python -m unittest discover -s tests/sparse_support -p 'test_cuda_training.py' -v` (14 explicitly skipped by the scheduler-admitted GPU gate)
  - `python -m compileall -q python/cellerator tests/sparse_support/test_transition_modes.py tests/sparse_support/test_cuda_training.py`
  - `git diff --check`
- Artifact identities: `_torch.cpython-313-x86_64-linux-gnu.so` at
  `/tmp/cellerator-transition-cuda-build/bindings/torch/` SHA-256
  `5be4fef31047be698d330a7f63c71d643838db4584db4a7adc32cb24cad00729`; CPU
  reference executable at
  `/tmp/cellerator-transition-cuda-build/tests/sparse_support/cellerator_sparse_support_numeric_reference`
  SHA-256 `75012b9c8f110fdcb9ae52b6e79722ffa7f7c729086e185580f07e9ccb31eadb`.
  Installed CPU wheel SHA-256
  `40377acfbc8e1030559dad52e4d7c7bcd1a12a436e552a3a4b0e4e39545ef6a7`.
- Remaining qualification: admitted GPU forward/VJP and checkpoint continuation,
  then Compute Sanitizer memcheck, racecheck, and synccheck. No performance claim
  is in scope.
- Final slot audit: v1 `apply(state,input,offsets,affected,A,B,bias,handle)`
  returns gradients at `[0,1,4,5,6]`; v2
  `apply(mode,state,input,offsets,affected,A,B,bias,handle)` returns gradients
  at `[1,2,5,6,7]`. v1 positions match the pre-change source at repository
  checkpoint `1d1dfda8c5b5ace00415c9bd0847f975f6b655b4`.
