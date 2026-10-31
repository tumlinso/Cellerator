# Trainable grouped-linear adapter

Recorded 2026-10-31 before implementation. This bounded work integrates the
existing SM70-capable cuBLAS operation with ordinary Torch-owned parameters.
It does not introduce a kernel, a parameter store, or a runtime dispatcher.

## Contract

`grouped_linear(inputs[N,I], weight[E,O,I], host_row_offsets[E+1],
implementation="auto") -> output[N,O]` computes each expert segment as
`inputs[start:end] @ weight[e].T`. Parameters, outputs, and gradients remain
Torch tensors. Empty segments and zero total rows are supported.

CPU auto execution uses a Torch reference. CUDA auto/native execution requires
the existing `_modeling_cuda` extension and supports matching FP16 or FP32
operands, FP32 accumulation, and first-order gradients. Reference execution
uses ordinary Torch differentiation and is independently selectable. No
implicit dtype conversions or device transfers occur. Noncontiguous inputs
are explicitly normalized through differentiable contiguous conversions.

## Existing implementation and selected approach

`GroupedLinear` in `include/Cellerator/modeling/cuda/grouped_linear.hh` stores
borrowed weight views and host offsets. The existing binding exposes forward,
backward, device views, and borrowed streams. A focused `autograd.Function`
will borrow each contiguous expert slice, allocate output and gradient storage
through Torch, and retain the executor and operands through the autograd tape.
Use the current Torch CUDA stream and allocator stream recording. Save original
contiguous operands through `save_for_backward` for mutation/version checks.
Native weight gradients are written directly into slices of one `[E,O,I]`
Tensor; the ordinary leaf receives that gradient without duplication.

One executor per invocation preserves immutable offsets and pointer bindings.
Its handle/preparation cost is part of operation cost; this is a correctness
integration, with no performance claim. The underlying implementation remains
library-backed FP16 tensor-core-capable cuBLAS GEMM, compatible with SM70.

## Risks and validation

Validate empty/unused experts, nonuniform segments, noncontiguous operands,
first-order VJPs, unsupported higher order, mutation before backward, shared
parameter accumulation, optimizer updates, and ordinary state dictionaries.
Native checks also cover FP16 represented-input error, FP32 master parameters
cast differentiably to FP16 compute, current non-default streams, and training
with ordinary layers around the custom operation. Compare against independent
Torch per-expert linear operations. CPU FP64 gradcheck supplements those checks.

CPU command:

```sh
PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests/framework -p test_nn_grouped_linear.py -v
```

GPU execution is limited to finite correctness jobs through authenticated Device
Scheduler, requesting one UUID-selected exclusive GPU and positive memory
accounting. No direct GPU probes or execution outside that authority. Preserve
job specifications, receipts, logs, extension identity and numerical results in
this workspace. No benchmark campaign is authorized.

Promotion requires CPU success plus admitted CUDA FP16/FP32 correctness and
end-to-end gradient/training evidence. Missing admission or toolchain capability
must be reported without substituting CPU results for GPU acceptance.

## Guidance consumed

- CUDA skill and focused Torch-extension/host-execution/Volta guidance under
  `/home/tumlinson/.agents/skills/cuda`.
- KDA agent entrypoint, basic-flow starter, and agent flow under
  `/home/tumlinson/kda`.
- Existing V100 qualification report and numerical grouped-linear implementation.

The root owns the project-level custom-op documentation registry. The operation
record below supplies its entry while avoiding changes outside assigned files.
