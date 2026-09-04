# Clean NVIDIA SDK acceptance

The NVIDIA acceptance configuration uses NVCC 12.9, GCC 12 as the CUDA host
compiler, and `CMAKE_CUDA_ARCHITECTURES=70` on Tesla V100-SXM2-16GB. It covers
profile-aware relations, generated and prelinked candidates, inline IR, custom
passes, graph/readiness contracts, the bounded direct-PTX experiment, and mixed
LTO. Exact-output checks precede comparison of preparation, transition, kernel,
epilogue, synchronization, and output-order cost components.

The SDK install exports the same compiler library and resources as host-only
mode; architecture selection remains runtime/build policy rather than stable ABI.
