# Volta is the design substrate

## Target and toolchain

The concrete target is V100, compute capability 7.0 (`sm_70`). Keep an sm_70-capable CUDA toolkit; the inspected repository route is CUDA 12.9. NVIDIA documents removal of offline compilation for pre-7.5 targets in CUDA 13. The package does not change the user's driver or toolkit. [S06]

The list below is a design palette, not a claim that this review audited every native opcode. The entire machine remains open for new ideas. For a new intrinsic/PTX form, check its exact target notes, compile it and inspect the emitted native instructions. PTX and SASS are not interchangeable vocabularies. [S02, S05]

## Available mechanisms to compose

Boolean/bitfield operations, packed integer arithmetic including DP4A, synchronized warp votes/shuffles/matches, ordinary floating arithmetic, FP16 Tensor Core operations, global/shared atomics, texture lookup and the register/shared/cache hierarchy are available design ingredients. Target legality is instruction-form-specific. [S01–S03]

Think in mechanisms, not API categories. A source predicate can become a mask; the mask supplies ranks; ranks route descriptors; descriptors fill a useful numerical tile; its output creates the next support mask. Or reverse the direction: a numerical query selects a support, the support selects a region, and that region's exact sequence answers the query.

Several lane assignments deserve separate experiments: lane-as-base, lane-as-packed-window, lane-as-predicate, lane-as-possible-state, lane-as-feature, lane-as-query and lane-as-hypothesis. The most elegant assignment may change between stages. Transposition and routing cost belong to the design, not to an invisible adapter.

## Do not accidentally design for a newer GPU

No Volta baseline assumption of `cp.async`, TMA, `ldmatrix`, warp `redux.sync`, BF16/TF32 arithmetic tiles, native integer/binary Tensor Core MMA, FP64 Tensor Core MMA, thread-block clusters or hardware sparsity. Ordinary DP4A and ordinary FP64 are different facilities. The documented FP16 `mma.m8n8k4` is a distinct optional low-level sm_70 path. [S02]

Do not take an Ampere/Hopper library's support claims as proof it runs efficiently—or compiles—on V100. A concept can be translated rather than abandoned, but the translated cost and synchronization must be explicit.

## Kernel rules that preserve meaning

Use explicit warp participation masks. A full-mask collective requires every named lane to reach it; tail items use neutral values or a consistent participation mask. Warp-uniform returns are different from lane-varying early returns. Independent thread scheduling makes old implicit lockstep assumptions unsafe. [S01]

For the supplied WMMA seed, pointers are 32-byte aligned, leading dimensions are 16, fragments are opaque, and all lanes participate uniformly. Zero padding describes absent rows/columns; keep validity and source IDs separately. Do not reinterpret an arbitrary register fragment as a stable semantic layout. [S03]

Atomics act on supported memory locations; there is no general “register atomic” mechanism. A reservation counter is not a publication protocol. The seed emitter is level-synchronous: consumers wait for the producer kernel on the same stream. Persistent queue variants need a separate progress, publication and termination design.

The finite-state and monomial kernels assume prevalidated states/permutations. All seed launches are small bounded fixtures with no allocation inside kernels. The maximum linear index must stay below 2^31 and all region-by-32 products must remain within that bound. The supplied harness uses tiny arrays. An implementation that generalizes the launch domain must add the appropriate host checks and indexing types.

## When lower-level work is worth trying

Inspect lower-level machinery when a useful idea looks awkward in normal CUDA. Also keep a plain implementation next to it. The previous lab showed why the semantic operator and its executor should be separable: an interesting affine mechanism can survive while CUB wins the scan implementation. [R13]

Unusual paths are explicitly welcome: texture interpolation as a small numerical surrogate, byte-dot routing, software double buffering, register butterflies, word-level carry circuits, bounded opcode cohorts and SASS-guided straight-line search. These are experiments, not declared optimizations. Texture interpolation is approximate; its coordinate precision belongs in the numerical contract. [S03]

No automatic fast-math flag is used by the standalone seeds. A later experiment may choose one, but must label the changed numerical mode rather than silently weakening an algebraic claim.

## Compile first without launching a GPU

The package's `tools/compile_cuda.sh` checks that `compute_70` is offered by the selected compiler, compiles the standalone harness, and optionally saves disassembly using its companion binary utility. It does not select a device or launch the executable. `CUDAHOSTCXX` can select an already available compatible host compiler; it does not install one.

Only run `bp_moon_cuda_smoke --run DEVICE` after the implementing environment has assigned that GPU. The harness is a few semantic checks, not a benchmark suite. Capture actual compiler, driver, device and binary identity in the receipt. CUDA compilation and execution were unavailable in the package authoring environment and are not claimed here.
