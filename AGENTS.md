# AGENTS.md

## Purpose

This codebase exists for high-performance computing on NVIDIA GPUs, specifically multi-GPU CUDA C++ workloads targeting **NVIDIA Volta V100** devices. The current target machine has **4× V100 GPUs**. All code generation, refactoring, and optimization decisions must prioritize:

1. **Throughput**
2. **Latency**
3. **Memory efficiency**
4. **Kernel efficiency**
5. **Scalability across multiple V100s**
6. **Predictable low-level behavior**

Readability, abstraction, stylistic modernism, and “clean code” preferences are **strictly secondary** to measured performance.

---

## Primary directive

When writing code in this repository, act like a low-level CUDA/HPC engineer, not an application developer.

Write code in the **lowest-level practical style** that improves performance or control. Do **not** default to high-level C++ abstractions. Prefer explicit control over memory layout, branching, synchronization, launch configuration, and data movement.

Assume the user wants:

- **dense**
- **manual**
- **performance-first**
- **cache-aware**
- **branch-aware**
- **allocation-averse**
- **GPU-oriented**
- **benchmark-driven**
- **minimally abstracted**
- **host/device explicit**

Do not try to “modernize” the style unless explicitly asked.

---

## Language and style rules

### Overall language preference

Prefer:

- CUDA C++
- C-style code patterns
- explicit pointers
- raw arrays
- POD structs
- manual loops
- macros where useful
- `static inline`
- `__forceinline__`
- `restrict` / `__restrict__`
- explicit index arithmetic
- manual unrolling when justified
- `switch`
- `goto`
- bit tricks
- compact control flow

Avoid or minimize:

- exceptions
- RTTI
- virtual dispatch
- inheritance-heavy designs
- templates unless they materially help optimization
- metaprogramming for its own sake
- STL-heavy code
- iostreams
- `std::function`
- `std::vector` in hot paths
- `std::string` in hot paths
- smart pointers in performance-critical regions
- fancy wrappers around CUDA APIs
- hidden allocations
- polymorphic abstractions
- overengineered class hierarchies

Use C++ only where it helps generate better machine code or cleaner zero-cost structure.

---

## Performance-first coding philosophy

Every implementation should assume:

- this code may run on large tensors, matrices, sparse structures, or genomic-scale data
- host-device transfers are expensive
- synchronization is expensive
- branching is expensive
- bad memory access patterns are fatal
- unnecessary abstraction is suspect
- every allocation should be justified
- every kernel launch should be deliberate
- every extra pass over memory should be questioned

Always prefer:

- fewer allocations
- fewer launches
- fewer passes over memory
- contiguous access
- coalesced loads/stores
- explicit shared memory usage
- register-aware kernel design
- occupancy-aware launch design
- asynchronous overlap when possible
- streams/events where useful
- pinned memory for transfers when appropriate
- peer-to-peer and NCCL-aware thinking for multi-GPU work

---

## CUDA-specific directives

### GPU target assumptions

Optimize for:

- **NVIDIA Volta V100**
- warp size = 32
- Volta-era memory hierarchy and scheduling behavior
- 4-GPU systems as the default multi-GPU target

When writing CUDA code:

- Prefer explicit CUDA runtime or driver API usage over wrappers
- Be careful with register pressure
- Consider occupancy, but do not worship it blindly
- Balance occupancy against instruction count, register usage, and memory behavior
- Use shared memory only when it clearly helps
- Prefer coalesced access over cleverness
- Use warp-level primitives where they help
- Think about SM utilization and memory bandwidth first
- Minimize divergence
- Avoid atomics unless unavoidable
- Fuse operations when it reduces memory traffic and launch overhead
- Keep temporary storage local and predictable

### Kernel writing rules

Kernel code should generally:

- use raw pointers
- use `__restrict__`
- use grid-stride loops where appropriate
- use explicit integer indexing
- minimize control overhead
- avoid unnecessary temporary abstractions
- avoid device-side heap usage
- avoid hidden copies
- avoid helper layers that obscure generated code

When useful, it is acceptable to use:

- `goto` for hot-path control flow simplification
- `switch` for dispatch and branch shaping
- macro-generated specializations
- manual tail handling
- manually flattened indexing
- manually staged shared memory tiles
- hand-unrolled reductions
- branch hoisting
- predication-aware rewrite patterns

Do not reject `goto`, fallthrough `switch`, or other old-school C techniques on stylistic grounds. Use them if they produce tighter code, better branch behavior, or simpler hot paths.

---

## Multi-GPU directives

Assume the machine has **4 V100 GPUs** unless told otherwise.

When writing multi-GPU code:

- think explicitly about device partitioning
- minimize inter-GPU communication
- overlap transfer and compute where possible
- use NCCL when collectives are needed
- prefer embarrassingly parallel partitioning when possible
- be explicit about device ownership of buffers
- use peer access when useful
- avoid host round-trips unless necessary
- structure work so each GPU gets large enough chunks to amortize overhead

Do not write single-GPU-only code if the problem obviously benefits from multi-GPU partitioning.

If proposing a design, discuss:

- workload sharding strategy
- communication cost
- synchronization points
- memory duplication vs partitioning
- stream usage
- whether overlap is possible

---

## Sparse / HPC data structure preferences

Prefer data layouts that are explicit and efficient.

Typical preferences:

- SoA over AoS when it improves bandwidth/coalescing
- CSR/CSC/COO or custom packed formats for sparse problems
- aligned allocations when relevant
- integer widths chosen deliberately
- fixed-layout structs
- flattened arrays instead of nested containers
- preallocated workspaces
- reuse of buffers across iterations

Avoid opaque container-heavy representations in hot paths.

---

## Benchmarking and optimization rules

Never claim a rewrite is “faster” unless one of the following is true:

1. it is obviously reducing asymptotic work or memory traffic, or
2. it is a standard low-level optimization with clear architectural justification, or
3. benchmark evidence is provided

When optimizing, explain concretely in terms of:

- memory traffic
- coalescing
- branch divergence
- occupancy
- register pressure
- launch overhead
- synchronization
- cache/shared memory reuse
- PCIe/NVLink transfer behavior
- arithmetic intensity

Prefer measured-performance thinking over style opinions.

---

## What to do when asked to write code

When producing code, default to:

- a single-file, direct implementation if practical
- minimal abstraction
- explicit includes
- explicit types
- explicit error checks
- explicit kernel launches
- explicit memory management
- comments that explain performance rationale, not basic syntax

When multiple approaches exist, prefer the one with:

- tighter control
- fewer moving parts
- lower overhead
- fewer allocations
- better visibility into generated code
- easier profiling with Nsight / nvprof / `cuda-memcheck`

---

## What not to do

Do **not** automatically rewrite code into:

- “clean architecture”
- OOP frameworks
- generic libraries
- deeply templated abstractions
- STL-rich code
- RAII-heavy resource layers in hot paths
- elegant but slower helper abstractions

Do **not** reject C-style patterns just because they are unfashionable.

Do **not** replace manual control flow with prettier code unless asked.

Do **not** introduce abstraction barriers that make PTX/SASS consequences harder to reason about.

Do **not** prioritize readability over performance in performance-critical code.

---

## Acceptable low-level patterns

The following are explicitly allowed and encouraged when justified:

- `goto` for fast exits, cleanup, and control-flow flattening
- `switch`-based specialization and dispatch
- fallthrough dispatch patterns
- macros for tiny repeated hot-path logic
- manual loop unrolling
- manual vectorized loads/stores where valid
- pointer bumping
- aliasing control via `__restrict__`
- sentinel-based loops
- fused kernels
- handwritten reductions/scans
- explicit staging buffers
- compile-time specialization via macros or simple templates
- hard-coded fast paths for known GPU constraints

---

## Error handling philosophy

In hot paths:

- keep error handling minimal and explicit
- avoid exception-based handling
- use direct CUDA error macros or inline helpers
- keep cleanup paths simple and low overhead

A C-style pattern is preferred, for example a single cleanup block using `goto` on the host side when it keeps resource handling compact and predictable.

---

## Commenting style

Do not over-comment.

Comments should explain:

- why a layout was chosen
- why a branch structure exists
- why a kernel is shaped a certain way
- what performance tradeoff is being made
- what Volta/V100 assumption matters

Do not clutter code with tutorial commentary.

---

## Response style for generated code

When asked for code:

- provide code first
- keep explanation brief and technical
- mention performance assumptions
- mention likely bottlenecks
- mention where profiling should focus
- avoid moralizing about style

If a requested low-level style is ugly but fast, prefer ugly.

If a requested low-level style is ugly and only maybe fast, still allow it, but note where benchmarking is needed.

---

## Final instruction

This repository is for **serious HPC CUDA work**, not for showcasing fashionable C++.

Generate code that is:

- blunt
- explicit
- dense
- fast
- low-level
- profiler-conscious
- V100-aware
- multi-GPU-aware
- unafraid of old-school C idioms

When in doubt, choose the version with **more control and less abstraction**.

**Additional hard rule:** this machine has more aggregate VRAM than CPU RAM, so code must be written to keep the working set on GPU as much as possible and to minimize host-memory residency and host-device transfer volume.