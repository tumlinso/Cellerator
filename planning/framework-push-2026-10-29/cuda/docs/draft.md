# CUDA runtime draft, 2026-10-29

Implement an optional CUDA C++ library and thin Python binding for owned streams,
events, device allocations, pinned host allocations, borrowed views, sparse
expert dispatch and feature profile quantization. Immutable structural indexes
remain separate from mutable values. FP16 storage and FP32 accumulation are the
primary numerical route, with explicit FP32 alternatives. NumPy host transfers
are explicit; Torch is not required.

The baseline is the host conditional dispatch/reference quantization APIs.
Runtime contracts require checked allocation arithmetic, same-device streams,
explicit asynchronous lifetimes, empty-input safety and ragged/rectangular
routing. Sparse outputs are compared to an independent CPU reference in the
worker smoke test; quantization has a host codebook oracle. No performance
claims or timings are planned. Current authority permits compilation, not GPU
execution, so smoke executables are built and left unexecuted pending scheduler
admission. Compile target is sm_70, CUDA12.9 nvcc with g++-12.

Plan: implement RAII resources; upload validated immutable plan/profile arrays;
connect independently owned sparse and quantization kernels; expose a lazy
optional Python adapter; compile the focused CMake library/binding and smoke
executables; record limits in docs/framework_cuda.qmd. Avoid legacy compiler
builds and environment repair. Exact configure command is recorded in evidence
once the focused harness exists.

Consumed guidance: cuda SKILL.md, native system guide, KDA agent-entry,
agent-flow and basic-flow read on 2026-10-29. No comparative candidate campaign.
