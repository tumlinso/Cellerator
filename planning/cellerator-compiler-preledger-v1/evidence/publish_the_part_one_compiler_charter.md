# Cellerator Compiler Part One charter

Todo: `CE-CCP1-A01-006`

## Mission

Cellerator Part One delivers a performance-governed, ahead-of-time compiler
family that turns biological organization and representative data into
executable organization. Cellerator owns source semantics, all public CEIR,
biological/profile reasoning, evidence discovery, exact certification,
composition grammar, basis and no-basis selection, program planning,
realization, lowering, compiler tooling, SDK compiler/runtime components, the
standard-library foundation, and celleratord.

The compiler must preserve biological identity, operation meaning, execution
order, structure/value/support generations, effects, persistence, and reuse
long enough to discover and select the fastest correct whole-program strategy.
Performance is judged by complete cost rather than aesthetic simplicity or an
isolated kernel result.

## Required Part One products

### `cellerator`

- A real NVCC-like Cellerator driver and reusable driver library.
- Ordinary C/C++ fallthrough through selectable GCC and Clang toolchains.
- File-local `#pragma cellerator` activation; filename is never semantic
  authority and NVCC never parses Cellerator syntax.
- Parsing and biological Sema for the specified small C++ extension.
- Explicit representative-profile participation for activated biological
  compilation, with no silently selected generic profile.
- Ahead-of-time CPU and NVIDIA object/executable production, including the
  validated sm70 path and bounded optional backend extensions.

### Public CEIR family

- Semantic IR for backend-independent biological meaning, types, effects,
  fields, identities, generations, control flow, and native boundaries.
- Planning IR for profile alternatives, discovered evidence, certified exact
  coverage, atoms, grammar, basis, decomposition, candidates, complete costs,
  constraints, selected or forced plans, and portable rulesets.
- Realization IR for chosen projections, packing, schedules, target contracts,
  generated/native operations, and backend lowering.
- Public mutable object models, canonical human-programmable text, sectioned
  binary artifacts, exact round trips, standalone compilation checkpoints,
  direct editing, inlining, reflection, extensions, replacement passes, and
  removable source-to-native provenance.

CEIR is Cellerator-native. LLVM IR, MLIR, generated C++/CUDA, PTX, and machine
code may implement later stages but do not replace the public CEIR family.

### Compiler-owned biological discovery and JBC rehoming

- Representative profile semantics and data-derived evidence.
- Proposal discovery and independent exact certification.
- Atom semantics and exact coverage; `atom` remains the authorized Part One
  term.
- Typed composition grammar, basis/no-basis, optional measured superatom
  promotion, decomposition, and partial-result algebra.
- Global operation/program IR, connected planning, portable schedule/ruleset
  compilation, candidate catalogs, external costs, selection, and realization.
- Useful existing JBC source, tests, and evidence moved, adapted, split, or
  wrapped with provenance. Equivalent code must not be silently rewritten or
  deleted before a no-code-loss receipt.

### `libCellerator`, standard library, and SDK

- Separately usable compiler and runtime/execution C and C++ APIs, plus a
  convenience umbrella rather than monolithic executable ownership.
- Direct supported runtime, operation, geometry, planner, prepared-program,
  session, readiness, and provider APIs for users who do not adopt the source
  language.
- A `.cell`-preferred standard-library foundation for higher biological
  constructions, containers, algorithms, and convenience layers; the base
  language contains only concepts the compiler intrinsically reasons about.
- Relocatable installation, componentized CMake packages, compiler/stdlib/
  profile/backend resources, and clean external consumers.

### `celleratord`

- Shared source, parser, AST, Sema, CEIR, profile, diagnostic, and query
  libraries with `cellerator`; no reverse engineering of a monolithic binary.
- Clangd-class ordinary C++ diagnostics, completion, hover, navigation, rename,
  symbols, and compile-command support.
- Cellerator syntax/semantic diagnostics plus profile, generation, field, IR,
  candidate, cost, decomposition, provenance, and source/native queries.
- Responsive incremental and cancellable behavior, including inline CEIR
  completion and validation.

## Programmer authority and low-level freedom

The easy path is automatic, but programmer authority proceeds through facts,
preferences, hard constraints, offered candidates/decompositions, forced plans,
writable CEIR, replacement passes, and manual/native realization. Cellerator
does not impose a closed biological ontology and does not remove ordinary
pointer, template, custom-layout, CUDA, or native-code control. Validators
diagnose aggressively while trusted, unsafe, unchecked, forced, and raw modes
remain available whenever the representation and backend can proceed.

## CellShard boundary

CellShard is a concrete downstream storage/application/runtime consumer. It
retains artifact storage, concrete dataset sharding and materialization,
staging, placement, residency, transport, leases, recovery, and runtime command
execution. It may supply external costs and capabilities through generic,
versioned interfaces. It does not own biological discovery, atom semantics,
grammar, basis, global program IR, planning, portable schedules, CEIR, or
compiler passes.

Part One may define and test only narrow ruleset export, materialization
requirements, external-cost callbacks, lowering resumption, and compatibility
seams needed to preserve current embedded CellShard consumers.

## Explicit Part Two deferral

Part One excludes general runtime Cellerator source compilation, a long-lived
JIT service, runtime specialization/recompilation, code-cache replacement,
distributed JIT coordination, and the associated runtime security/deployment
model. Bounded same-compilation compiler-prelude transforms and ordinary driver
loading of emitted PTX are Part One techniques, not a claim of general JIT.

Part One also excludes deep CellShard ruleset-driven materialization for
arbitrary datasets, new atom-store/runtime formats beyond migration needs,
fleet placement, residency/cache/lease evolution, transport/RDMA/object-store
delivery, production distributed execution, runtime schedule recovery, and
deep JIT integration. None may become a hidden prerequisite for Part One
completion.

## Build and acceptance boundary

The root build becomes CXX-first with CUDA selection `AUTO`, `ON`, or `OFF`;
the approved default decision is `AUTO`. Frontend, CEIR, profiles, diagnostics,
compiler SDK APIs, and relevant celleratord functions must build without CUDA,
while accelerator-aware planning and NVIDIA validation remain mandatory when
the declared resources are available.

Part One closes only after milestones M00 through M90 are reached,
`CE-CCP1-I41-PART1-COMPLETE` is frozen, useful JBC code is preserved, and the
final capability matrix proves the compiler, CEIR, profiles, planning,
realization, reflection/passes, AOT CPU/NVIDIA paths, SDK, standard library,
celleratord, provenance, tests, sanitizers, fuzzing, performance review, and
Part Two separation.

## Source reconciliation

This charter was checked against four current language/IR documents:

1. `docs/language/cellerator-language-specification.md`;
2. `docs/language/cellerator-programming-guide.md`;
3. `planning/cellerator-compiler-preledger-v1/06_SOURCE_LANGUAGE_IMPLEMENTATION_PLAN.md`;
4. `planning/cellerator-compiler-preledger-v1/07_SEMANTIC_IR_IMPLEMENTATION_PLAN.md`.

Ownership and completion were additionally checked against
`02_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md`,
`03_CELLERATOR_CELLSHARD_SUPERSESSION_AND_JBC_REHOMING.md`,
`19_PART_ONE_COMPLETION_AND_ACCEPTANCE.md`,
`20_DEFERRED_PART_TWO_INVENTORY.md`, and
`24_SPECIFICATION_RECONCILIATION.md`. The active plan decisions retain the
`atom` name and select CUDA configuration `auto`. Where older language text
permits JIT-assisted compilation or implies broader global IR ownership, this
Part One charter applies the explicit bounded-prelude and CellShard-seam
reconciliation above.
