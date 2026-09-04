# Part One release and bootstrap reproducibility bundle

The release baseline starts from integrated M80 source revision
`2f6f1eda215047d9164d9cc93a510b842af3512e`. The companion SHA-256 manifest
pins a reference profile, CEIR specification/example, SDK consumer, build
matrix, benchmark contract, and JBC provenance. Focused J03 gates and the host
and NVIDIA configure/install commands in their receipts complete the bundle.

Reproduction uses CMake's explicit `CELLERATOR_ENABLE_CUDA=OFF` host path or
NVCC 12.9/GCC 12 with `CMAKE_CUDA_ARCHITECTURES=70`. Benchmark commands remain
governed by the repository benchmark mutex and GPU reservation policy.
