# Volta design envelope

Primary references: R1–R4 and R15–R16 in research/SOURCES.md. This is an SM70 experiment, not an invitation to import later architecture instructions.

## Available design materials

Use FP16-input Tensor Core contractions with FP32 accumulators, ordinary FP32/FP16 arithmetic, integer/address operations, warp ballot/shuffle/match collectives, popcount/bit manipulation, shared memory, caches, atomics/reductions and texture sampling where appropriate. Volta can issue integer and floating-point work through distinct execution resources; this is an opportunity to stage addressing alongside arithmetic, not a guarantee of perfect overlap [R1].

PTX is a virtual ISA lowered to target code. The delivered raw-MMA primitive follows a documented PTX register layout. Inspect generated SASS for spills, redundant moves and actual HMMA emission, but do not infer instruction latency/throughput from the high-level template alone.

## Do not borrow unavailable features

No `cp.async`, `ldmatrix`, TF32, BF16 Tensor Core mode, FP8, 2:4 sparse MMA, binary MMA, WGMMA, TMA or CTA clusters in an SM70 path. BF16 can be manipulated as encoded data in other ways, but not claimed as native V100 Tensor Core arithmetic. A header compiling for a newer device does not make that instruction available on Volta.

Use the established CUDA12.x toolchain. CUDA13 removed offline compilation/library support for Volta [R4]. The payload CMake intentionally rejects CUDA13 for its optional SM70 build; it does not demand any change to the project's installed toolkit.

## Correct collective execution

WMMA calls are warp-collective; all required lanes must execute with compatible arguments. Memory pointers/strides must satisfy WMMA alignment rules. Raw PTX layouts and opaque WMMA fragments are distinct interfaces [R2,R3].

`match_any_sync`/`shfl_sync` move or compare register values. They do not replace memory barriers. Specify participants and never read from a shuffle source that did not participate. Under independent thread scheduling, old implicit lockstep assumptions are insufficient [R1,R3].

In the patch seed, inactive tail warps return uniformly and active warps use only their own shared-memory slice plus `__syncwarp`; there is no later CTA-wide barrier. A different implementation with CTA-wide reuse must reconsider those returns.

## What to inspect lightly

Compile the three kernels for `sm_70`, execute the supplied mapping smoke, then add one same-policy output and gradient witness per chosen path. Inspect `ptxas` register/spill output and optionally disassemble. Do not start a giant kernel sweep before the first model is callable.

Performance counters are optional observations, not completion gates for every idea. Never report bandwidth, occupancy or speedup that has not been measured. Record unsupported shapes and precision instead of burying them behind fallback success.
