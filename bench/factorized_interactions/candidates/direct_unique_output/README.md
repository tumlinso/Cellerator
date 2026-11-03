# Unique-output direct candidate

This benchmark-only candidate specializes the factorized gated-product law
when every output target has exactly one contributing mechanism. It selects
only for a nonempty invocation whose host-prepared metadata has
`max_output_mechanisms == 1`, `mechanism_count == extents.outputs`, and whose
direct route is accepted by the frozen numerical backend query. Host-prepared
structure is trusted under the same contract as the native direct entrypoints.

For that case, each `(mechanism, channel tile, cell tile)` CTA owns a disjoint
output region. The warp lanes traverse contiguous cells, and up to four warps
cover channels. Forward writes the final FP16 or FP32 output directly and saves
the same two-plane FP32 gate/slope tape. It does not allocate or initialize the
FP32 output scratch, issue output atomics, or launch the reduction/cast pass.
Backward intentionally delegates to the frozen native direct VJP, preserving
its zero-safe product derivatives, tied-parameter reductions, and stable
fallback behavior.

Other shapes and any invocation selected for the stable backend call the
native direct implementation unchanged. `DirectUniqueOutputBackend` reports
the actual backend (`direct_unique_output`, `direct`, or `stable`) so consumers
can distinguish this specialization from delegated paths. No performance claim
is made until correctness qualification and paired measurements are run through
the assigned Autobench/Scheduler and KDA experiment workflow.

The candidate consumes the shared contract in
`planning/interaction-layout-proposal-2026-11-02/CONTRACT.md`, the public native
types in `include/Cellerator/modeling/cuda/factorized_interaction.hh`, and the
exact compensated gate helper in
`src/modeling/cuda/factorized_interaction_numerics.cuh`. Its pointer, alignment,
device, capacity, overflow, and alias guards mirror the frozen direct forward
guard in `src/modeling/cuda/factorized_interaction_direct.cu`; this duplication
is isolated here because the task forbids modifying the captured core while it
qualifies.

## Compile-only check

From the repository root with CUDA 12.9:

```sh
nvcc -std=c++17 -O3 -arch=sm_70 -lineinfo -Xptxas=-v \
  -Iinclude -Isrc/modeling/cuda \
  -c bench/factorized_interactions/candidates/direct_unique_output/direct_unique_output.cu \
  -o /tmp/direct_unique_output_sm70.o
```

This compiles the candidate translation unit only. Linking requires the existing
native factorized-interaction CUDA target. Do not infer device correctness or
performance from compilation.
