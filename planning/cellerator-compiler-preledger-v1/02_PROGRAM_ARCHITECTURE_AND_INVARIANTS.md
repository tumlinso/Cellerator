# Program architecture and invariants

## Compiler thesis

Cellerator compiles biological organization into executable organization.

The compiler does not merely call a fixed library routine. It preserves typed domains, axes, relations, support, logical edge identity, ordering, values, generations, effects, persistence, representative states, and operation meaning long enough to discover reusable structure and choose physical execution.

```text
C/C++ plus Cellerator source
    -> source and C++ semantic integration
    -> Semantic IR
    -> representative profile environments
    -> evidence discovery and exact certification
    -> Planning IR search space
    -> grammar, basis, decomposition, candidates, complete cost
    -> selected portable ruleset
    -> Realization IR
    -> generated C++ / CUDA / LLVM / PTX / native provider calls
    -> ordinary object files and executables
```

CellShard sits downstream as an application/storage/runtime consumer of compiled rules for concrete data instances.

## Public programming continuum

```text
ordinary C/C++
    -> Cellerator semantic source
    -> planning facts and constraints
    -> explicit candidates and decomposition
    -> writable Semantic IR
    -> writable Planning IR
    -> writable Realization IR
    -> generated/native backend code
```

No layer is a ceremonial dump. All public CEIR levels are writable and inlineable. The easy path remains automatic.

## Major architectural resolutions

- No permanent Clang fork. Upstream Clang libraries are wrapped behind a versioned adapter for C++ parsing, Sema, templates, constexpr, and source tooling.
- The driver is NVCC-like. It owns Cellerator semantics and delegates conventional compilation to discovered, overridable backends.
- `#pragma cellerator` is file-local and extension-independent.
- `.cell` is preferred, not semantic authority.
- Activated biological compilation requires representative profile input.
- The root build becomes CXX-first with optional CUDA AUTO/ON/OFF.
- Compiler/tooling internals may use a modern C++ standard while legacy CUDA/runtime code remains in explicit compatibility islands.
- Three public CEIR levels remain the primary family. Internal checkpoints are facets or resumable artifacts, not separate public languages.
- Cellerator LTO is explicit. It does not absorb arbitrary plain C++ across fields or translation units.
- Same-compilation transforms use a bounded compiler-prelude staging model rather than requiring a general Part Two JIT.
- Validation distinguishes structural impossibility from compiler disagreement. Unsafe continuation remains available where technically representable.
- Identity and provenance are cold/compile-time unless execution actually requires them.

## Durable invariants

| ID | Rule |
| --- | --- |
| `CE-CCP1-INV-ONE-AUTHORITY` | Part One uses the existing Cellerator Todo authority only, including coordinated migration work under components/CellShard; historical authorities are preserved, not rewritten. |
| `CE-CCP1-INV-COMPILER-AUTHORITY` | Cellerator owns discovery and compilation of reusable rules from biological structure and representative data, including evidence, certification, atoms, grammar, basis, program IR, schedules, planning, CEIR, and lowering. |
| `CE-CCP1-INV-CELLSHARD-APPLICATION` | CellShard remains a storage/application/runtime built on Cellerator; it may materialize and serve concrete instances but does not own biological compiler discovery, grammar, basis, global program IR, or portable schedule compilation. |
| `CE-CCP1-INV-PRESERVE-JBC` | Useful JBC implementation and tests are moved, adapted, split, or wrapped with provenance; no equivalent reimplementation or deletion occurs before a no-code-loss migration receipt. |
| `CE-CCP1-INV-HISTORY` | Historical Todo runs, commits, branches, interfaces, and evidence remain reconstructable; supersession is additive and never edits historical results. |
| `CE-CCP1-INV-ATOM-NAME` | The term atom remains the canonical name during Part One. Any rename requires explicit human approval and a separate migration decision. |
| `CE-CCP1-INV-TRADITIONAL-LAYOUT` | Compiler code follows the existing traditional include/Cellerator/compiler, src/compiler, tools, tests, bench, stdlib, profiles, cmake, and docs organization rather than creating a permanent compiler fork repository. |
| `CE-CCP1-INV-CENTRAL-INTEGRATION` | Root CMake, umbrella headers, central registries, generated manifests, package exports, documentation authority, and the CellShard gitlink are integration-lane owned; leaf providers publish isolated fragments. |
| `CE-CCP1-INV-SDK-COMPONENTS` | libCellerator is a coherent SDK composed of separately usable compiler and runtime/execution components plus a convenience umbrella; no monolithic executable owns reusable logic. |
| `CE-CCP1-INV-HOST-ONLY` | Frontend, CEIR, profiles, diagnostics, libCellerator compiler APIs, and relevant celleratord functionality configure and build without CUDA; accelerator backends remain optional build components. |
| `CE-CCP1-INV-ACCELERATOR-CENTRAL` | Optional CUDA does not make Cellerator CPU-first: accelerator-aware planning and NVIDIA backend validation remain mandatory Part One capabilities when corresponding resources are available. |
| `CE-CCP1-INV-NO-CLANG-FORK` | Cellerator is its own driver/frontend and may use upstream Clang libraries or clangd processes through versioned adapters; no permanent Clang fork is the architectural foundation. |
| `CE-CCP1-INV-BACKEND-AGNOSTIC` | Semantic IR is independent of GCC, Clang, NVCC, LLVM, and PTX; target and toolchain specificity increases only in Planning and Realization IR and backend layers. |
| `CE-CCP1-INV-ORDINARY-OBJECTS` | AOT compilation produces ordinary platform object and executable artifacts; CEIR companion metadata may be embedded or sidecar but does not require a bespoke universal executable container. |
| `CE-CCP1-INV-PRAGMA-FILE-LOCAL` | #pragma cellerator enables the language only from the directive to the end of that physical file/include instance and never leaks through include boundaries; file extension is not the semantic switch. |
| `CE-CCP1-INV-CXX-FALLTHROUGH` | Unactivated ordinary C/C++ follows the selected downstream compiler with no Cellerator semantic compilation; activated source remains interoperable with ordinary C/C++ code and opaque/effect-contracted native calls. |
| `CE-CCP1-INV-SMALL-LANGUAGE` | Only information intrinsically reasoned about by the compiler belongs in the base language; higher constructions are Cellerator standard-library code written in .cell where practical. |
| `CE-CCP1-INV-BIOLOGICAL-IDENTITY` | Shape never establishes biological equivalence. Domain, axis, order, geometry, partition, relation, logical edge, structure epoch, value generation, and support generation remain explicit where relevant. |
| `CE-CCP1-INV-TYPED-OPS` | Numeric/storage/compute/accumulation/output types and operation semantics participate before candidate selection; half, bf16, float, and double may legitimately produce different operations and realizations. |
| `CE-CCP1-INV-LOW-LEVEL-FREEDOM` | Cellerator remains low-level and explicit. Compiler-semantic types prevent meaningful errors but do not impose a closed biological ontology or remove ordinary pointer, template, custom layout, and native-code control. |
| `CE-CCP1-INV-EXPLICIT-FIELDS` | Execution fields explicitly authorize joint biological planning; optimization does not silently cross field, C++ observable-effect, or uncontracted native-call boundaries. |
| `CE-CCP1-INV-PROFILE-REQUIRED` | Activated biological semantic compilation requires an explicitly bound representative profile environment. Pure C++ fallthrough and structural CEIR tooling are exempt; generic reference profiles are never silently selected. |
| `CE-CCP1-INV-DATA-DERIVED-PROFILES` | Representative profiles are data-derived compiler evidence optimized for fast expressive analysis; they do not encode hand-authored execution policy or concrete runtime pointers. |
| `CE-CCP1-INV-CONTROL-HIERARCHY` | Programmer authority progresses from automatic optimization through facts, preferences, hard constraints, custom candidates/decompositions, forced plans, writable IR, replacement passes, and manual/native realization. |
| `CE-CCP1-INV-PUBLIC-WRITABLE-IR` | Semantic, Planning, and Realization IR are public, human-readable, textually programmable, directly compilable, writable, reflectable, inlineable, extensible, and replaceable programming surfaces. |
| `CE-CCP1-INV-CEIR-NATIVE` | CEIR belongs to Cellerator. LLVM IR, MLIR, generated C++, CUDA, PTX, and machine code may implement or follow it but never substitute for the public Cellerator IR family. |
| `CE-CCP1-INV-EXACT-COVERAGE` | Approximate evidence may propose atoms and decompositions, but execution admission requires independently certified exact coverage, contribution ownership, canonical recovery, and compatible partial-result algebra. |
| `CE-CCP1-INV-COMPLETE-COST` | Compiler decisions expose complete cost including compilation, preparation, conversion, packing, transfer, memory, launches, synchronization, execution, residuals, canonicalization, and reuse amortization where relevant. |
| `CE-CCP1-INV-HOT-COLD-SEPARATION` | Cold IR, provenance, profile, discovery, planning, and artifact metadata never force runtime parsing, discovery, allocation, global sorting, or pointer-rich metadata into sealed hot execution paths. |
| `CE-CCP1-INV-EXPERT-CONTROL` | Every major compiler mechanism remains accessible to expert code, including profiles, decomposition, candidates, costs, passes, physical layout, stage graphs, native operations, and direct PTX where technically meaningful. |
| `CE-CCP1-INV-NO-PATERNALISM` | Validators diagnose aggressively, but trusted, unsafe, unchecked, forced, and raw continuation remain available whenever the representation and backend can technically proceed; only structural impossibility must hard-fail. |
| `CE-CCP1-INV-PROVENANCE-COLD` | Source-to-native provenance is recoverable through compile-time sidecars or removable artifact metadata and imposes no mandatory hot-path runtime overhead. |
| `CE-CCP1-INV-EXPLICIT-MECHANICS` | Diagnostics, reflection, and IR expose inferred biology, profile state, generations, alternatives, costs, rejections, decomposition, realization, and backend output instead of presenting compiler decisions as magic. |
| `CE-CCP1-INV-SHARED-COMPILER-LIBS` | cellerator and celleratord share source, parser, AST, Sema, CEIR, profile, diagnostics, and query libraries; celleratord does not reverse-engineer a monolithic compiler executable. |
| `CE-CCP1-INV-RUNTIME-ACCESS` | Programmers who never adopt the Cellerator language retain direct supported access to useful runtime, operation, geometry, planner, prepared-program, session, readiness, and provider APIs. |
| `CE-CCP1-INV-STDLIB-CELL` | The standard-library foundation is compiled Cellerator source with .cell preferred. Base language semantics are not hidden as ordinary classes merely for implementation convenience. |
| `CE-CCP1-INV-NEGATIVE-RESULTS` | Experimental candidates, backends, discovery strategies, and optimizations may validly complete as evaluated-not-promoted when exactness and complete-cost evidence do not justify promotion. |
| `CE-CCP1-INV-PERFORMANCE` | Performance is the governing criterion, measured at complete compiler and execution cost rather than aesthetic simplicity or isolated kernel time; explicit low-level complexity is acceptable when progressively exposed. |
| `CE-CCP1-INV-SPEC-MUTABLE` | Current language and IR documents guide implementation but may be reconciled when source evidence or validated implementation exposes contradictions; performance, control, modern architecture, and explicit semantics govern. |
| `CE-CCP1-INV-PART-TWO-DEFERRED` | General JIT/runtime compilation and deep CellShard materialization, persistence, placement, residency, transport, and runtime evolution remain Part Two except for narrow versioned seams required by Part One. |
| `CE-CCP1-INV-NO-NVCC-PARSE` | NVCC never parses Cellerator syntax. Cellerator owns source semantics and emits generated CUDA/C++ or lower artifacts for NVCC actions. |
