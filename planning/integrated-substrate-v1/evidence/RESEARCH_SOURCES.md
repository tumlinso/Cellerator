# Primary technical references

Accessed 2 October 2026. These constrain machine claims or identify prior art; the new cards' equations are derived in this package and checked by its references. No paper's performance result is claimed for these repositories.

- R01: NVIDIA, PTX ISA 8.8, CUDA 12.9.1 archive. https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html — target-specific instruction support, shuffle participation, MMA semantics and integer/bit machinery. A manual containing an instruction does not establish SM70 support.
- R02: NVIDIA, Volta Tuning Guide 12.9. https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html — Volta execution, independent thread scheduling and memory/resource constraints.
- R03: NVIDIA, CUDA 13.0 Update 3 Release Notes. https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html — CUDA 13.0 removed offline compilation/library support for Maxwell, Pascal and Volta; keep a verified CUDA 12.x SM70 builder.
- R04: Willsey et al., egg: Fast and Extensible Equality Saturation. https://arxiv.org/abs/2004.03082 — a primary implementation precedent for typed/analysed equality-saturation work, not a required new dependency.
- R05: Dao et al., Monarch: Expressive Structured Matrices for Efficient and Accurate Training. https://arxiv.org/abs/2204.00595 — useful structured-matrix family precedent; not a justification for treating every constrained factorization as an exact arbitrary dense operator.
- R06: NVIDIA, CUDA C++ Programming Guide 12.6. https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/ — synchronization, tensor-core interfaces and numerical/execution contracts.
- R07: NVIDIA, cuBLAS 12.9.2. https://docs.nvidia.com/cuda/archive/12.9.2/cublas/index.html — baseline numerical library and explicit compute/math policy.

The two already installed predecessor moonshot catalogues contain broader literature searches. Preserve those exact source lists and cards at their original paths. This successor concentrates new research on integrating their mechanisms and on the additional derived compositions, rather than repeating a broad catalogue search.
