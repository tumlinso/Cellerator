# Quad micro MMA

`quad_mma.cuh` declares native CUDA entry points in
`cellerator::experimental::moonshot`. Each panel is an independent row-major
8x4 by 4x8 product. One warp executes four independent PTX
`mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` products. Missing tail
panels load deterministic zeros and suppress writes. Panel/thread arithmetic
uses 64 bits before multiplication; panel counts remain uint32.

`launch_quad_mma_checked(QuadMmaRequest, stream)` accepts explicit capacities
in elements. It rejects insufficient capacities, misalignment, address overflow,
output overlap with either input, architectures below SM70, excessive grid
extent, allocations outside the current device, and streams on another device.
Inputs may overlap each other. Nonempty buffers must be CUDA device allocations;
managed and host buffers are unsupported. The compatibility `launch_quad_mma`
wrapper asserts minimum capacities from the panel count; callers needing verified
capacity metadata should use the checked request. No wrapper allocates or
synchronizes. Borrowed buffer lifetime and generation matching remain caller
responsibilities because this primitive carries no biological state metadata.

Numerical policy: stored FP16 operands, FP32 accumulation, FP32 outputs, zero
initial accumulator. No input quantization or nonlinear stage is added. Tests
compare against FP32 products of the stored half values, reject nonfinite
results, and use absolute tolerance 1e-4 for the exact dyadic fixtures. This
fixture tolerance does not establish an error bound for arbitrary data.
Nonfinite input values are not scanned or sanitized. Derivatives and performance
are not measured by this primitive.

Build from the repository root:

```sh
/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc -std=c++17 -arch=sm_70 experiments/moonshot-parallel-v1/mma/quad/quad_mma.cu experiments/moonshot-parallel-v1/mma/quad/quad_test.cu -o /tmp/moon_quad_test
python3 experiments/moonshot-parallel-v1/mma/quad/check_mapping.py
/tmp/moon_quad_test --host
```

CUDA 12.9 is used for the stream-device admission query. Parent CMake may add
this directory after enabling CUDA. Targets are `moonshot_quad` and
`moonshot_quad_test`; the registered CTest runs only host admission checks.

With an assigned GPU resource lease, run `/tmp/moon_quad_test`. It exercises
counts 1, 3, 4, 7, 16, 17 with panel-distinct operands and 32 one-hot K/column
fixtures per count. Each fixture checks the full output and a four-float tail
guard: 198 fixtures and 101376 output values. No timing claim is made.

Initial evidence: CUDA 12.9 SM70 compilation passed; host admission and pure
Python fragment ownership/mapping checks passed. GPU execution is pending the
root's resource grant. Implementation remains uncommitted for root acceptance.
