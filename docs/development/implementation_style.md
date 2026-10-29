# Cellerator Implementation Style

This file defines local implementation style. Architecture and ownership are governed by `AGENTS.md`, `scope.md`, and `docs/architecture.qmd`.

## Organize by behavior

Split implementation around independently testable behavior:

- structure construction;
- value binding;
- projection construction;
- planning;
- persistent preprocessing;
- launch binding;
- kernel execution;
- epilogue;
- validation;
- instrumentation.

Do not place all stages of a workflow in one translation unit merely because one public function invokes them.

Files above roughly 600 lines deserve review. Files above roughly 1000 lines should normally be split unless a generated source, external ABI, or tightly coupled kernel family makes separation more costly.

## Keep cost ownership visible

A function that allocates, synchronizes, converts, reorders, hashes, constructs descriptors, launches kernels, or transfers memory should make that behavior visible in its name, contract, or call site.

Avoid constructors or convenience wrappers that perform expensive hidden work.

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
- repeated `std::vector` growth in copy-sensitive paths;
- per-entry heap allocation;
- hidden container conversions;
- unversioned serialized C++ object graphs.

RAII is preferred when it preserves pointer stability and explicit cost. Raw ownership is not a performance feature.

## Hot-path rules

Steady-state execution should normally contain no:

- allocation;
- descriptor creation;
- structure hashing;
- host synchronization;
- per-tile host dispatch;
- canonicalization without a consumer requirement;
- format conversion not represented in the plan.

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

## Local READMEs

Directory READMEs explain only:

- what the directory owns;
- which central contracts it implements;
- which files are transitional;
- where tests and benchmarks live.

They must link to the authoritative architecture documents rather than restating a competing architecture.
