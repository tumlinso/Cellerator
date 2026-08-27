# CelleraTorch native-training autograd adapter

`native_training_autograd` is a thin adapter over Cellerator's frozen
`training_program` contract. It does not implement backward math, allocate a
parameter store, select a kernel, or create a CUDA stream.

The current native training program is deliberately a combined N=16 step:

```text
forward -> epilogue -> explicit-transpose backward
        -> sparse-value update -> bias update -> readiness publication
```

Consequently the independent CelleraTorch forward operation supplies a
detached Torch-visible forward result. During Torch backward, this adapter runs
the combined native step once on the current Torch CUDA stream, returns the
native dense-input gradient, and records the native result metadata. The
recomputed native forward output is caller-provided scratch and is not silently
substituted for the already-visible forward result.

## Ownership and gradients

The binding owns no Cellerator storage. Its weak lifetime token must be backed
by the native owner of the program, value plane, bias, workspace, readiness
records, and other launch-bound pointers. Forward and backward reject an
expired token before dereferencing native state.

Cellerator applies the relation-value and bias updates. Those canonical native
parameters receive no Torch gradient tensors and must not also be passed to a
Torch optimizer. Torch receives only the dense-input gradient. The last native
result exposes the two updated parameter descriptors, consumed/published
generations, readiness record, and completion stream for inspection.

## Runtime contract

- Only contiguous CUDA float32 rank-two tensors with width 16 are accepted.
- Tensor shape, stride, device, and native view metadata must agree exactly.
- The visible forward output must be detached from any other autograd path so
  backward cannot accidentally execute two training systems.
- Backward uses `at::cuda::getCurrentCUDAStream` for the tensor device. It does
  not create a private stream, call `cudaSetDevice`, or synchronize the host or
  device.
- Cellerator's readiness check governs stale, missing, and cross-stream value
  generations. A successful enqueue is not treated as global completion.
- Each autograd context may execute backward only once, including when a caller
  retains the graph.

General-N training, Torch-owned parameter optimization, separable native
forward/backward entry points, and model wrappers remain unsupported.

## Validation

The leaf test compiles directly against the frozen native libraries because
shared CelleraTorch CMake/package registration belongs to CE-LIVE-43. It uses
one Tesla V100-SXM2-16GB, CUDA 12.9, driver 580.173.02, sm_70 code, and the
source-built `/usr/local` libtorch C++11 ABI.

The test compares one Torch autograd step with a direct Cellerator step over
the same topology, tensors, output gradient, parameters, generation, and
non-default current stream. It then consumes generation 2 from a different
Torch stream and publishes generation 3 without rebuilding the prepared
program. Negative cases cover CPU/wrong-dtype/wrong-shape tensors, unsupported
N, competing or missing `requires_grad`, missing/mismatched native parameters,
unready generation, repeated backward, and expired native lifetime.

- Foreground correctness evidence:
  `850c10c3-670d-4d53-99b8-c2c71467bb98`.
- Compute Sanitizer memcheck evidence:
  `2883d87c-be4b-4197-b4db-4b45d36f7c55`.

Both controller runs returned zero with an uncontaminated single-GPU lease.
No timing or performance claim is made by CE-LIVE-42.
