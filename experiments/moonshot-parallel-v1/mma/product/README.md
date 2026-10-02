# Product2/JVP prototype

`product.hpp` exposes nonowning FP32 arrays in namespace
`cellerator::experimental::moonshot::product`. `prepare_product2` validates
shape, pointer alignment, output overlap, current device, stream device, actual
allocation extents, SM70, and all live indices. Preparation copies the index
arrays to host and synchronizes the supplied stream. `launch_product2` checks
the recorded generations and stream, then launches asynchronously. Empty work
accepts empty descriptors without accessing CUDA.

The caller retains buffers and keeps index arrays immutable through completion.
Structural, value, activity, and parameter generations bind the input snapshot.
Actual allocation bounds use `cuMemGetAddressRange`; memory kinds unsupported by
that query are rejected. Device memory from `cudaMalloc` is the qualified test
route. Managed memory is rejected. The prototype accepts SM70 only.

One mechanism produces `(k*x[a])*x[b]` and
`k*fmaf(v[a],x[b],x[a]*v[b])`. Repeated arguments keep both derivative terms;
argument order is preserved. Outputs are mechanism-local scratch. There is no
assembly, VJP, parameter gradient, second-order derivative, or performance claim.
`UINT32_MAX` is internal padding only: live sentinel arguments are rejected.
All warp lanes participate in match/shuffle operations, including final padding.

CPU-only target `moon_product_oracle` checks counts 0, 1, 31, 32, 33, 127, and
129 against independent double-precision algebra and finite differences.
CUDA target `moon_product_smoke` runs host metadata admission checks by default;
`moon_product_smoke --gpu` also checks those counts, output guard values,
zero/repeated/ordered arguments, index bounds, sentinel, host-memory rejection,
actual allocation extents, aliases, and stale snapshots. GPU execution requires
the root's resource grant.

Compile independently with CUDA 12.9:

```sh
nvcc -std=c++17 -arch=sm_70 --fmad=false product.cu tests.cu -lcuda -o product_smoke
./product_smoke
# Under an assigned GPU lease:
./product_smoke --gpu
```

CMake links `CUDA::cudart` and `CUDA::cuda_driver`. C++ CPU checks require no CUDA.
