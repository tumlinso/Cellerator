# Research and decision log

Research clarifies hardware and runtime contracts; it does not override the architectural review or justify unmeasured speed claims. No external ML framework or algorithm is introduced as an implementation dependency by this package. Source-derived observations are separately listed in `07_SOURCE_LEDGER.md`.

## Primary sources

### R1: NVIDIA CUDA Programming Guide, C/C++ Language Extensions

https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html

WMMA requires aligned pointers, admissible leading dimensions and cooperative warp behavior. Repair legality of the actual called path, including derived subpointers; logical width alone is not a physical stride.

### R2: NVIDIA CUDA 12.0 Runtime API, Event Management

https://docs.nvidia.com/cuda/archive/12.0.0/cuda-runtime-api/group__CUDART__EVENT.html

A recorded event captures preceding stream work; subsequent re-recording and previously submitted waits require careful lifetime reasoning. The reader-done protocol is our design derived from these rules, not a feature automatically supplied by a generation counter.

### R3: NVIDIA CUDA 12.8.1 Runtime API, Stream synchronization behavior

https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/stream-sync-behavior.html

Use nonblocking test streams and explicit dependencies so legacy default-stream synchronization cannot mask a missing edge.

### R4: NVIDIA CUDA 13.0.3 Release Notes

https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html

CUDA 13 removed offline compilation/library support for Volta. This epic uses a supported CUDA 12.x sm70 toolchain rather than an unqualified latest-toolkit requirement.

### R5: Shi et al., FlashSparse: Minimizing Computation Redundancy for Fast Sparse Matrix Multiplications on Tensor Cores

https://arxiv.org/abs/2412.11007

Research motivation to account for redundant dense work and data movement instead of promoting a path solely because it uses Tensor Cores. Its reported target results are not V100 predictions and are not used as Cellerator acceptance.

### R6: NVIDIA CUDA 12.6.3 Math API, half conversion functions

https://docs.nvidia.com/cuda/archive/12.6.3/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html

Nearest-even conversion is explicit in operand packing and value storage. The package reference checks storage rounding separately from the continuous derivative.


## Decisions made during planning

**Retain the full-f32 gradient path and add explicit half-rounded operands.** Existing relation execution consumes f32 states, whereas the contraction provider consumes half. A precision conversion must be a visible policy rather than an unnoticed wiring artifact. A small f32 sparse VJP also provides a useful independent execution comparison. This does not open arbitrary precision support.

**Use persistent edge order, not automatically identical forward/gradient tiles.** Keep one mutable value authority and maps; gradient WMMA panels may have a different useful geometry. Extract only support scores directly into that authority's order. Do not expand all gradient channels or materialize a global dense relation.

**Separate producer readiness and reader retirement.** A generation-ready event proves values exist, not that a remote stream has finished reading them. One bounded const lease keeps the requested cross-stream consumption useful without adding concurrent prepared execution or immutable snapshots. Multiple sequential consumers can use different streams; overlapping leases and concurrent host calls remain out of contract.

**Repair and execute WMMA, but measure promotion.** This epic includes both the concrete legality repair and a real prepared mixed path. An eligible forced-hybrid test must show actual WMMA and residual launches. Sparse wins are legitimate measurements, not excuses to hide the WMMA implementation. More general Tensor Core layout/provider work is an explicit near-term deferred item.

**Test the real program from both origins.** Compiler-originated descriptors must come from parsed source and verified effects, then reach the same physical execution. Copying a native descriptor into a wrapper called compiler is not acceptance. The example's strings are bounded source slices, not a claim about full .cell compilation.

**Keep learning coupled to biological relation structure.** The small synthetic regulatory target makes differentiation/update understandable. It does not assert that this toy model is a novel biological method. Future learning belongs in Cellerator when its biological semantics enable useful new methods or performance, rather than duplicating generic dense infrastructure.

**34 leaf tasks, not a task quota.** Contract, runtime, CUDA, parser and independent acceptance responsibilities have different failure modes. Within those groups related edits are bundled. There are no per-file bookkeeping tasks and no new broad campaign merely to reach 50.

**No accelerator speedup threshold invented.** Required outcomes are mathematical correctness, actual WMMA reachability, persistent reuse, explicit lifetime accounting, and honest candidate comparison. A small synthetic dense fixture cannot prove an application-wide biological speedup. I06 records enough phase data to choose or defer default promotion.
