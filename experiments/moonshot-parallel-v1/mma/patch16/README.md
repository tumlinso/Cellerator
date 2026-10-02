# Patch16 forward primitive

This lane-local adaptation retains the immutable seed's two WMMA products:
`T = FP32(L_half X_half)`, `V = FP16_RN(tanh(T))`, `Y = FP32(V R_half)`.
Every patch is an independent contiguous row-major 16×16 matrix. Four warps
handle four patches per block. Partial blocks return whole inactive warps;
no block-wide barrier is used. Forward only; derivatives remain unimplemented.

`patch16.cuh` publishes `launch_patch16(L,X,R,Y,count,stream)` in
`cellerator::experimental::moonshot`. Its raw launcher checks null/alignment
and trusts preparation. `patch16::validate(Request)` checks supplied extents,
capacities, sm70 architecture, alignment, range arithmetic and output aliases.
`patch16::launch_prepared(Request,stream)` additionally checks current device,
actual architecture, stream device and device-allocation pointer provenance.
Zero patches are a no-op after metadata validation. Inputs may overlap each
other; outputs cannot overlap an input. Pointer capacities are caller-provided,
not inferred from CUDA allocation bounds. Dtype is fixed by the C++ signature;
this standalone primitive has no generation or semantic-state metadata.

Build (without GPU execution):

```sh
cmake -S experiments/moonshot-parallel-v1/mma/patch16 -B /tmp/moonshot-patch16-build \
  -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc
cmake --build /tmp/moonshot-patch16-build -j2
ctest --test-dir /tmp/moonshot-patch16-build -R patch16_admission --output-on-failure
```

Controller-authorized GPU correctness:

```sh
/tmp/moonshot-patch16-build/patch16_smoke
```

The smoke returns 77 when CUDA or sm70 is unavailable. It checks counts 0,1,4,5,
nonsymmetric, identity and saturating inputs, finite outputs, a host same-stored-half
oracle, and an untouched extra output patch. Absolute/relative tolerance is
0.003 + 0.002|reference|, allowing nonlinear half-rounding boundary differences.
The CPU admission test does not query CUDA or execute GPU work. Compilation and
host admission evidence live in `evidence/`. Current GPU correctness qualification
requires the aggregate `../verify_gpu_evidence.py` gate and its fresh build
manifest in `../results/`; retained historical runs alone do not qualify source.
PTX inspection records both WMMA products. ptxas reports 32 registers, 6144 bytes
shared memory, and zero spill loads/stores. No performance claim is made.
