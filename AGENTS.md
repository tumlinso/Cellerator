# AGENTS.md

## Purpose

This repository is for high-performance numerical computing on **4× NVIDIA Volta V100 GPUs**.

Everything here is performance work. The priorities are:

1. throughput
2. latency
3. memory bandwidth efficiency
4. GPU residency
5. multi-GPU scaling
6. predictable low-level behavior

Readability, elegance, modern C++ style, and abstraction quality are not priorities unless they are free.

---

## Non-negotiable style rules

Write code like a low-level CUDA/HPC engineer.

Do not introduce abstraction for its own sake. Do not hide memory movement, indexing, launches, synchronization, or ownership behind wrappers, classes, helper frameworks, or “clean” interfaces.

### Hard bans

The following are not to be used:

- `std::vector`
- smart pointers of any kind
- exceptions
- RTTI
- virtual functions in performance-sensitive code
- inheritance-heavy design
- `std::function`
- iostreams in hot or infrastructure paths
- container-heavy STL code
- RAII-heavy resource wrappers
- generic helper layers that hide CUDA, NCCL, or memory behavior
- “modern C++” refactors that make machine behavior less obvious

If dynamic memory is needed, use explicit allocation and explicit cleanup.

If ownership matters, make ownership visible in the code.

If a wrapper obscures cost, do not use it.

---

## Primary directive

Prefer direct, dense, explicit code.

Use:

- CUDA C++
- raw pointers
- flat arrays
- POD structs
- manual loops
- explicit indexing
- explicit kernel launches
- explicit transfers
- explicit synchronization
- `__restrict__`
- `static inline`
- `__forceinline__`
- macros where useful
- `switch`
- `goto`
- fallthrough dispatch
- manual specialization
- handwritten fast paths

Old C style is acceptable and encouraged when it produces tighter or more controllable code.

Do not remove ugly low-level code just because it is ugly.

---

## Host memory policy

This machine has **more aggregate VRAM than CPU RAM**.

That means host memory is not the main working set. Host RAM is a staging and control layer only.

### Rules

- keep data on GPU as long as possible
- keep intermediates on GPU
- minimize host-side copies
- minimize device-host-device round trips
- do not materialize large host mirrors unless absolutely necessary
- do not design pipelines around CPU memory convenience

Prefer:

- GPU-resident pipelines
- preallocated device buffers
- reused device workspaces
- fused kernels
- device-to-device transfers
- NCCL collectives
- peer-to-peer transfers
- chunked streaming when necessary
- pinned memory only when unavoidable

Avoid:

- duplicate host copies
- CPU-owned intermediates
- host preprocessing that could run on device
- spilling back to host between GPU stages

The CPU should orchestrate. The GPUs should hold the data.

---

## Toolchain policy

Use the lowest-level practical tool that does the job without abstraction pollution.

### Preferred order

1. raw CUDA C++ when control matters
2. NVIDIA HPC toolkit libraries when they are a legitimate numerical fit
3. OpenACC where it maps cleanly and does not degrade control, residency, or performance

### NVIDIA HPC toolkit policy

Everything in NVIDIA’s HPC toolkit is allowed **if it is legitimately the right tool** and not abstraction pollution.

That includes, when appropriate:

- cuBLAS
- cuSPARSE
- cuFFT
- NCCL
- CUB
- Thrust only in limited cases where it is genuinely the best low-level practical choice and does not infect hot paths with abstraction-heavy style
- OpenACC
- other NVIDIA HPC SDK components that directly serve performance

Do not avoid NVIDIA libraries on ideological grounds.

Do not wrap them in useless abstraction.

Do not force the codebase into a library-shaped architecture just because a library is being used.

Use them when they are numerically appropriate, operationally direct, and performance-legitimate.

Do not use them when a custom kernel would clearly avoid extra passes, extra conversions, extra synchronization, or extra transfers.

---

## CUDA policy

Optimize for **Volta V100**.

Always think in terms of:

- memory traffic
- coalescing
- divergence
- register pressure
- launch overhead
- synchronization cost
- arithmetic intensity
- occupancy as a tradeoff, not a religion

Kernel code should usually have:

- raw pointer arguments
- `__restrict__`
- flat indexing
- grid-stride loops where appropriate
- minimal branching
- minimal helper layers
- no hidden allocations
- no device-side heap use unless absolutely unavoidable

Allowed and encouraged when justified:

- `goto`
- `switch`
- fallthrough dispatch
- manual unrolling
- manual tail handling
- pointer bumping
- flattened control flow
- handwritten reductions
- handwritten scans
- macros for tiny hot-path specializations
- separate kernels for materially different cases

Do not reject these patterns because they look old.

---

## OpenACC policy

OpenACC is allowed where it is genuinely useful.

Good use cases:

- regular dense numerical loops
- structured kernels
- regular map/reduce-like operations
- code that benefits from acceleration without needing hand-written CUDA kernels

Rules for OpenACC:

- control data regions explicitly
- keep data resident across regions
- minimize copies
- do not let OpenACC drag code back into a host-centric model
- do not use OpenACC where raw CUDA is clearly required for low-level control

If OpenACC introduces transfer noise, poor residency, or weak execution control, do not use it.

---

## Multi-GPU policy

Assume **4× V100 GPUs with NVLink**.

When writing multi-GPU code:

- shard work explicitly
- minimize communication volume
- minimize synchronization points
- overlap communication and compute where possible
- prefer large chunks of work per GPU
- use **NCCL** for collectives by default
- use peer access where useful
- prefer direct GPU-to-GPU exchange over host-mediated transfers
- do not bounce inter-GPU data through host memory unless absolutely necessary

Be explicit about:

- ownership of buffers
- sharding strategy
- communication pattern
- stream layout
- overlap opportunities
- duplication versus partitioning
- when NCCL is used and why

Do not write single-GPU-minded code for problems that clearly want sharding.

---

## Data structure policy

Prefer machine-friendly data layouts.

Usually that means:

- flat arrays
- SoA over AoS when it improves bandwidth or coalescing
- CSR / CSC / COO for sparse data where appropriate
- direct packed layouts
- fixed-layout structs
- deliberate integer widths
- aligned storage where useful
- reusable scratch buffers
- preallocated workspaces

Do not use container-heavy representations in hot paths.

---

## Benchmarking and optimization policy

Do not call something “faster” unless:

1. it obviously reduces work or memory traffic, or
2. it obviously reduces launches, transfers, or synchronization, or
3. it is benchmarked

Explain optimizations in terms of:

- bytes moved
- access regularity
- coalescing
- divergence
- register usage
- occupancy tradeoffs
- launch count
- synchronization count
- transfer volume
- NVLink / PCIe behavior
- GPU residency

Do not justify rewrites by saying they are cleaner, nicer, safer, or more maintainable unless that cost is literally zero.

---

## Code generation rules

When asked to write code:

- write direct implementations
- keep abstractions near zero
- make ownership explicit
- make memory movement explicit
- make synchronization explicit
- make launch configuration explicit
- comment performance decisions, not beginner syntax

Do not “improve” code by making it more object-oriented, more generic, or more modern.

If two versions are possible, choose the one with:

- less abstraction
- less hidden cost
- less host-memory use
- fewer allocations
- fewer passes over memory
- more predictable machine behavior
- easier profiling in Nsight

---

## Explicitly allowed patterns

These are explicitly allowed and should not be removed on style grounds:

- `goto`
- `switch`
- fallthrough
- macros
- pointer arithmetic
- manual unrolling
- fused kernels
- handwritten reductions
- handwritten scans
- specialized fast paths
- direct library calls
- C-style cleanup blocks
- hard-coded dispatch for materially different cases

---

## Error handling policy

Keep error handling explicit and cheap.

Prefer:

- direct CUDA error checks
- direct status checks for NVIDIA libraries
- inline macros or tiny helpers
- one cleanup block with `goto` when useful

Do not use exception-based error handling.

---

## Interconnect policy

The GPUs in this machine are connected with **NVLink**.

Treat GPU-to-GPU communication as a first-class path. Do not default to host-mediated transfer patterns when inter-device exchange is needed.

Rules:

- prefer GPU-to-GPU communication over routing through host memory
- use **NCCL** for collective communication
- use peer-to-peer transfers where appropriate
- design multi-GPU work assuming fast direct interconnect is available
- avoid host bounce buffers for cross-GPU exchange unless there is no better option
- overlap NVLink communication with compute where possible

When writing multi-GPU code, assume NCCL is the default collective layer unless there is a specific reason not to use it.

---

## Final instruction

This is a **performance repository**, not a software architecture exercise.

Write code that is:

- blunt
- explicit
- dense
- low-level
- GPU-first
- multi-GPU-aware
- comfortable with ugly C idioms
- willing to use NVIDIA HPC toolkit components when they are truly the right tool
- hostile to abstraction pollution
- hostile to host-memory-centric design

Never use `std::vector`.

Assume NVLink is available and NCCL should be used for collective multi-GPU work by default.

Never use smart pointers.

Never hide machine-relevant behavior behind pretty code.

When in doubt, choose the version with less abstraction and more control.