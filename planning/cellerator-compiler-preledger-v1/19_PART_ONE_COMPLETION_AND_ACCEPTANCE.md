# Part One completion and acceptance

The root program does not close when one parser demo runs.

Part One closes only at `CE-CCP1-MILESTONE-M90`, after `CE-CCP1-I41-PART1-COMPLETE` is published and the final capability matrix proves all required surfaces.

## Required production surface

### Compiler and language

- real `cellerator` executable and reusable driver library;
- ordinary C++ fallthrough through GCC and Clang;
- file-local `#pragma cellerator`;
- real parsing and biological Sema for the specified language;
- representative profile participation and missing-profile error;
- typed relations, operation families, fields, effects, persistence, reuse, generations, controls, native interoperability;
- standard `.cell` library foundation.

### Public CEIR

- Semantic, Planning, and Realization IR object models;
- human-programmable canonical text and sectioned binary form;
- exact CEIR text round trip;
- standalone compilation from each checkpoint;
- multiple profile alternatives;
- direct editing, inlining, reflection, and provenance;
- extension preservation;
- trusted/unsafe/unchecked paths.

### Compiler planning

- migrated JBC evidence discovery and exact certification;
- atoms, grammar, basis/no-basis, superatoms if promoted;
- global program IR and portable rulesets under Cellerator ownership;
- decomposition, partial algebra, candidates, external costs, connected planning;
- complete-cost reports, rejection reasons, deterministic fallback;
- no lost useful JBC implementation or tests.

### Realization and backends

- ordinary CPU object path;
- NVIDIA/NVCC object path on sm70;
- optional Clang CUDA/NVPTX/direct PTX boundaries and at least one direct-PTX demonstration;
- selected cover, projections, packing, persistent order, stage graph, readiness, generations, graph capture, runtime bindings;
- ordinary linkable objects/executables.

### Programmable compiler

- source reflection of fields, operations, profiles, and all IR levels;
- inline Semantic/Planning/Realization IR;
- custom and replacement passes;
- custom ops/types/attributes/extensions;
- bounded same-compilation self-transform;
- verified/checked/trusted/unsafe/unchecked behavior;
- source-to-native provenance and explainability.

### Program scope and SDK

- object CEIR metadata;
- explicit cross-TU import and Cellerator LTO;
- libCellerator C/C++ compiler APIs;
- direct runtime/execution APIs remain usable;
- relocatable install and CMake package;
- compiler/stdlib/profile/backend resources;
- clean external consumers.

### celleratord

- ordinary clangd-class C++ diagnostics, completion, hover, navigation, rename, symbols, and compile commands;
- Cellerator syntax and semantic diagnostics;
- profile, generation, field, IR, candidate, cost, decomposition, and source-native queries;
- inline CEIR completion/validation;
- responsive incremental/cancellable behavior.

## Required validation

- host-only clean build/install/consumer matrix;
- NVIDIA clean build/install/execution matrix;
- source, CEIR, profile, planner, backend, LTO, pass, SDK, and LSP tests;
- fuzzing and sanitizer receipts;
- complete-cost benchmark report;
- documentation examples compile and run;
- JBC migration and history receipts;
- Part Two dependency audit.

## Final workstream

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-J03-001` | Integrate central compiler targets and registries | Merge isolated source fragments into root/subsystem CMake, grammar/dialect/backend/pass registries, umbrellas, generated manifests, standard-library resources, and package exports under one integration owner. |
| `CE-CCP1-J03-002` | Integrate JBC migration and CellShard compatibility | Advance the embedded CellShard gitlink only after source/provenance receipts, compile compatibility adapters, and prove all compiler semantics are Cellerator-owned while concrete storage/runtime remains usable. |
| `CE-CCP1-J03-003` | Reconcile language specification with implementation | Update normative syntax, nesting, profiles, effects, control hierarchy, C++ integration, errors, and implementation-defined behavior to match validated compiler behavior and governing philosophy. |
| `CE-CCP1-J03-004` | Reconcile IR specification with implementation | Update Semantic/Planning/Realization IR, reflection, inline IR, passes, staging, extensions, trust modes, native boundary, round-trip, and LTO rules. |
| `CE-CCP1-J03-005` | Reconcile programming guides and examples | Update developer-facing source and IR guides from minimal programs through profiles, planning, realization, custom passes, unsafe native paths, LTO, SDK, and celleratord. |
| `CE-CCP1-J03-006` | Publish architecture and migration completion records | Document final directory layout, Cellerator/CellShard ownership, preserved JBC provenance, superseded charters, interfaces, build modes, backends, and Part Two seam. |
| `CE-CCP1-J03-007` | Run clean host-only SDK acceptance | From a clean checkout with CUDA disabled, build/install cellerator, libCellerator, celleratord, stdlib, profiles, and package metadata; compile ordinary C++, `. |
| `CE-CCP1-J03-008` | Run clean NVIDIA SDK acceptance | Build/install with NVCC on sm70, compile and execute profile-aware relation programs, generated/prelinked candidates, inline IR, custom pass, graph/readiness, direct PTX experiment, and mixed LTO. |
| `CE-CCP1-J03-009` | Validate all final Part One capabilities | Check every acceptance condition: real driver, pragma parsing, profiles, all IR levels writable, CEIR round-trip/input, reflection, passes, self-transform, unsafe paths, CPU/NVIDIA objects, toolchain overrides, JBC migration, LTO, SDK, stdlib, celleratord, provenance. |
| `CE-CCP1-J03-010` | Audit deferred Part Two separation | Prove full JIT/runtime compilation and deep CellShard materialization/runtime evolution are not hidden prerequisites; retain only versioned seams and explicit deferred records. |
| `CE-CCP1-J03-011` | Run final performance and regression review | Compare current runtime/provider baselines, compiler overhead budgets, generated execution, planning quality, object sizes, and editor latency. |
| `CE-CCP1-J03-012` | Create release and bootstrap reproducibility bundle | Package source revisions, profile fixtures, CEIR examples, toolchain manifests, build presets, tests, benchmark commands, SDK consumer projects, and provenance needed to reproduce acceptance. |
| `CE-CCP1-J03-013` | Freeze Part One completion checkpoint | Close only after all mandatory interfaces/checkpoints, final host and NVIDIA acceptance, documentation reconciliation, SDK installation, celleratord baseline, JBC preservation, and deferred boundary checks pass. |
