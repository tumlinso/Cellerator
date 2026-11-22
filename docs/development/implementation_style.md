# Cellerator Implementation Style

The protected root `coding_style.md`, published as [C++ coding style](../coding_style.qmd), is the canonical C++ style authority. Read it before C++ design, implementation, or style review. This guide supplements it with file organization and subsystem conventions. Architecture and ownership are governed by `AGENTS.md`, `scope.md`, and `docs/architecture.qmd`.

## Organize by behavior

Keep a coherent calculation, its traversal, and its invariants easy to inspect
together. Split files or helpers for a real ownership, compilation, reusable
computation, or independently useful contract boundary. Structure construction,
preparation, execution, validation, and instrumentation are possible boundaries,
not a required file/class/stage hierarchy. There are no file- or function-length
quotas. Do not fragment direct numerical code to satisfy a line count or add
forwarding layers merely to give every phase a separate name.

Ordinary host calculations and common instantiations belong in compiled source.
Public templates/device helpers remain visible when specialization, CUDA
compilation, or consumer customization requires it. Review the complete code a
caller must understand, including helpers and data structures.

## Keep cost ownership visible

A function that allocates, synchronizes, converts, reorders, hashes, constructs descriptors, launches kernels, or transfers memory should make that behavior visible in its name, contract, or call site.

Owners and member functions must expose real resource or invariant contracts.
Only explicit provisioning acquires/grows/releases backing storage. Reuse
sufficient capacity without allocation, copying, or transfer; numerical
preparation and evaluation consume supplied storage. Keep cheap stack views
cheap. The root style's allocation, metadata, lifetime, and reuse requirements
apply to host planners and resource owners as well as GPU execution.

## CUDA helpers and kernels

Use private inline device helpers for reusable lane-local, warp-local, or element-local operations that do not own traversal or launch policy.

Use standalone kernels for:

- traversal ownership;
- memory-staging policy;
- cross-row or cross-feature reductions;
- tile classification;
- projection construction;
- fused operations whose register, shared-memory, or occupancy behavior must be measured.

Keep architecture-specific implementations behind stable contracts. Do not force one source-level schedule across Volta, Ampere, Hopper, and Blackwell.

## Data representation

Prefer:

- structure-of-arrays for columnar access;
- contiguous, aligned sections;
- narrow local indices where validated;
- pointer-free persistent images;
- caller- or session-owned buffers;
- explicit capacity and identity;
- compile-time or prepared dispatch in hot paths.

Avoid:

- pointer forests;
- STL machinery or equivalent generic growing machinery in authored native production;
- per-entry heap allocation;
- hidden container conversions;
- unversioned serialized C++ object graphs.

Use Cellerator owners with explicit provisioning and borrowed views. RAII may
express actual resource release; it must preserve pointer/lifetime contracts and
make any synchronization cost explicit. Neither smart-pointer machinery nor raw
ownership is an automatic performance or style justification. Necessary foreign
API types remain at their adapter boundary under the root style's exceptions.

## Hot-path rules

Repeated execution within provisioned capacity must perform zero allocations
or backing-storage growth, including metadata and ownership bookkeeping. Reuse
prepared descriptors, structure hashes, ordering, and topology while their
documented dependencies remain valid. Cheap borrowed stack descriptors are
permitted. Rebuild only affected state when dependencies change, including
data-derived support, membership, or eligibility.

Host waits, per-tile host dispatch, canonicalization, conversion, and repacking
need a concrete operation/consumer requirement and an explicit cost boundary.
The root allocation policy also governs preparation, capacity reuse, and
completion-safe scratch reuse. Do not add a pool/cache/allocator framework merely
to implement these rules.

Fuse passes when fusion removes material traffic or launches and does not create a larger loss through spills, occupancy collapse, or duplicated work.

Keep an unfused reference path when it materially improves testing, diagnosis, or planner comparison.

## Naming

Use `snake_case` for files, functions, variables, POD structures, and CLI flags.

Name contracts by their semantics rather than the first backend that implements them.

Examples:

- `relation_structure`, not `csr_matrix_contract`;
- `feature_major_projection`, not `custom_spmm_v2`;
- `launch_bindings`, not `runtime_request`;
- `value_generation`, not `cache_version`.

Backend-specific names belong in backend-specific files.

## Comments

Comments should explain:

- identity and lifetime;
- ordering;
- ownership;
- bounds;
- numerical semantics;
- why a representation or synchronization boundary exists;
- measured reasons for a non-obvious optimization.

Do not use comments to declare an implementation "fast" without a benchmark reference.

## Scoped source check and acceptance

Pin the review base before editing: use the pre-change commit for local work or
the agreed branch base for a multi-commit change. Run from the repository root:

```sh
python3 tools/checks/check_no_inappropriate_core_stl.py --base <review-base>
```

Use that same base after staging and committing. After a commit, `--base HEAD`
would hide its added lines and is not validation of that change. The scoped
check covers added production lines, including new/untracked source files;
existing debt is not permission for new violations. Omit `--base` only for a
deliberate full-tree debt audit. Do not expand legacy allowances to pass a new
change, and do not make a full legacy cleanup a prerequisite for scoped work.

This lexical check cannot prove absence of allocation, aliases, hidden library
work, numerical opacity, or bad abstractions. Review the affected computation
and its helpers against the root rules. For allocation-sensitive implementation
changes, use focused first-use/repeated/value-change/within-capacity checks and
record allocations, peak/retained bytes, and relevant copies/transfers. Reuse
existing checks/instrumentation and keep validation proportionate to the change.

## Local READMEs

Directory READMEs explain only:

- what the directory owns;
- which central contracts it implements;
- which files are transitional;
- where tests and benchmarks live.

They must link to the authoritative architecture documents rather than restating a competing architecture.
