# Source-language implementation plan

## Implementation target

The source language remains a small but semantically heavy extension of C++.

```cpp
#pragma cellerator

<[
    response = expression -[regulation]-> genes;
]>
```

The pragma enables grammar for the rest of that physical file. The execution field opens one explicit semantic planning region. Relation application preserves biological connectivity without fixing storage or kernel shape.

The implementation must cover the full current language specification, not only these two constructs.

## Base-language boundary

Base language concepts are limited to facts the compiler intrinsically reasons about:

- domains, axes, states, relations, support and order;
- structure, value, and active-support generations;
- operation kinds and output effects;
- execution fields;
- representative profile binding;
- persistence/reuse and mutation/effect semantics;
- planning facts, preferences, constraints, candidates, forcing;
- IR reflection, inline IR, passes, and native fragments.

Higher constructions, containers, algorithms, and friendly biological wrappers belong in the `.cell` standard library.

## Nested fields

This plan resolves nesting as follows:

- an inner field inherits outer profile facts and defaults;
- inner constraints overlay inherited policy;
- the inner field is a separately nameable planning subproblem and optimization boundary;
- the outer planner sees the inner field’s declared effects and result contract;
- explicit semantic-field inlining may dissolve the boundary;
- no implicit cross-boundary fusion occurs.

This is simple enough to implement and preserves explicit coder authority.

## Missing representative data

Activated biological semantic compilation without any bound representative profile is an error.

Exceptions are:

- pure C++ fallthrough;
- parse/Sema-only tooling that explicitly does not claim data-aware compilation;
- standalone structural CEIR editing;
- explicit use of a shipped generic reference profile.

Generic reference profiles emit a low-performance/testing warning and are never silently selected.

## Control hierarchy

```text
automatic compiler search
    -> planning facts
    -> preferences/objectives
    -> hard constraints
    -> user candidate/decomposition
    -> forced plan
    -> writable CEIR / replacement passes
    -> manual native realization
```

## Workstream task catalog

### C01: grammar and parser

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-C01-001` | Freeze the executable grammar revision and token vocabulary | Convert the reconciled language specification into an explicit versioned grammar, token kinds, contextual keywords, precedence table, and extension points. |
| `CE-CCP1-C01-002` | Implement parser cursor and bounded lookahead | Build a token cursor over activated source islands with checkpoint/rollback, balanced delimiter tracking, and deterministic recovery without reparsing the whole translation unit. |
| `CE-CCP1-C01-003` | Parse compiler-semantic declarations | Parse domain, axis, state, relation, support/order, profile, field, candidate, pass, and IR-binding declarations defined by the reconciled specification. |
| `CE-CCP1-C01-004` | Parse biological type constructors and qualifiers | Parse typed domain endpoints, value/storage/compute/accumulation types, orientation, mutability, ordering, generation, persistence, and human biological tags without resolving them. |
| `CE-CCP1-C01-005` | Parse anonymous execution fields | Parse <[ . |
| `CE-CCP1-C01-006` | Parse named execution fields and references | Parse named field declarations, calls/references, explicit export/import intent, and field-level policy blocks needed for reflection and cross-TU authorization. |
| `CE-CCP1-C01-007` | Parse relation application | Parse source -[relation]-> destination with typed captures, orientation modifiers, result/update forms, and composable expression placement without lowering it to a storage primitive. |
| `CE-CCP1-C01-008` | Parse non-relation operation families | Parse transpose, support contraction, segmented reductions/normalization, edge map/gate, sparse axis update, bundles, chains, moments, hierarchy pool/broadcast, and exchange through coherent operation syntax selected by the language spec. |
| `CE-CCP1-C01-009` | Parse planning facts, preferences, and hard constraints | Parse field-local and operation-local facts for profiles, reuse, persistence, budgets, objectives, target classes, candidate inclusion/exclusion, and forced realization while preserving hierarchy of authority. |
| `CE-CCP1-C01-010` | Parse effects, mutation, generations, and epochs | Parse native effect contracts, structure/value/support/order mutations, publication, epoch boundaries, generation assertions, and expert identity manipulation. |
| `CE-CCP1-C01-011` | Parse inline CEIR blocks | Parse semantic, planning, and realization inline IR regions with typed captures, results, nesting, validation mode, and abstraction transitions. |
| `CE-CCP1-C01-012` | Parse reflection and compiler-transform constructs | Parse IR reflection, pass declarations, pipeline insertion/replacement, compiler preludes, same-compilation transform application, and compile-time IR construction. |
| `CE-CCP1-C01-013` | Parse native/backend fragments | Parse typed generated-C++, CUDA, PTX, and raw-native blocks with explicit target, inputs, outputs, clobbers/effects, and fallback requirements. |
| `CE-CCP1-C01-014` | Implement structured parser recovery | Recover at field, declaration, operation, qualifier, and inline-IR boundaries; emit one primary diagnostic and bounded notes rather than cascades. |
| `CE-CCP1-C01-015` | Expose parser library and parse-tree dump APIs | Publish reusable parser entrypoints, immutable parse trees, visitors, and deterministic text/JSON dumps for compiler, tests, and celleratord. |
| `CE-CCP1-C01-016` | Deliver full grammar conformance | Parse every normative and provisional syntax example in docs/language, mark intentionally changed examples, and produce a grammar coverage matrix. |

### C02: AST and source diagnostics

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-C02-001` | Freeze AST node ownership and lifetime | Use arena-owned immutable nodes with stable compile-time handles, explicit parent/region links, and no runtime burden in emitted programs. |
| `CE-CCP1-C02-002` | Define source-level AST node families | Represent declarations, fields, operations, policies, effects, profile bindings, inline IR, reflection, passes, and native fragments without mirroring parser productions mechanically. |
| `CE-CCP1-C02-003` | Bind C++ AST references safely | Store adapter-owned stable references to resolved C++ declarations, expressions, types, templates, and constants without exporting raw Clang pointers through public APIs. |
| `CE-CCP1-C02-004` | Implement Cellerator symbol tables and scopes | Resolve domains, axes, relations, fields, profiles, candidates, passes, IR names, and imported program symbols with C++ namespace context where appropriate. |
| `CE-CCP1-C02-005` | Assign deterministic source identities | Derive compilation-local stable IDs from semantic owner, canonical source location, declaration identity, and revision, while separating persistent user identities from transient AST handles. |
| `CE-CCP1-C02-006` | Preserve token and macro provenance | Attach definition, expansion, physical file, shadow placeholder, and generated-source mappings as cold sidecars. |
| `CE-CCP1-C02-007` | Implement AST visitors, matchers, and queries | Provide allocation-free iteration views and indexed lookup for fields, relations, operations, effects, and source positions. |
| `CE-CCP1-C02-008` | Create structured frontend diagnostic records | Represent severity, category, source ranges, notes, fix-its, related symbols, compiler phase, and stable diagnostic ID independently of terminal rendering. |
| `CE-CCP1-C02-009` | Implement source-aware fix-its | Generate edits for missing pragma, malformed field delimiters, relation endpoint mismatches, absent profile bindings, effect-contract omissions, and deprecated syntax. |
| `CE-CCP1-C02-010` | Create deterministic AST dump and snapshot formats | Provide human text and machine JSON snapshots that include semantics and source identities but omit unstable raw addresses. |
| `CE-CCP1-C02-011` | Support incremental AST identity reuse | Reuse unchanged file/field/subtree identities across celleratord reparses using content and dependency hashes, with conservative invalidation across macros/templates. |
| `CE-CCP1-C02-012` | Freeze the AST and source-diagnostics interface | Publish the public AST/query/diagnostic contracts and demonstrate original source to parse tree to resolved C++ capture mapping. |

### C03: biological types and operations

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-C03-001` | Freeze compiler-semantic type categories | Define language-level domain, axis, state, relation, support, order, structure, value plane, profile state, field, candidate, and IR handle types only where the compiler performs intrinsic reasoning. |
| `CE-CCP1-C03-002` | Implement domain and human biological tag semantics | Separate abstract domain identity from optional tags such as gene, cell, locus, enhancer, read, chromosome, population, and trajectory. |
| `CE-CCP1-C03-003` | Implement axis semantics | Bind domain, global extent, logical order, geometry, partition, local extent, and recovery identity as distinct properties. |
| `CE-CCP1-C03-004` | Implement state semantics | Type state by axis/domain, element/storage type, feature width, order, residency intent, mutability, and generation class while retaining ordinary pointer/view interoperability. |
| `CE-CCP1-C03-005` | Implement relation endpoint semantics | Bind source and destination axes/domains, stable relation/structure identity, logical edge identity, support, order, orientation, value plane, and mutation policy. |
| `CE-CCP1-C03-006` | Implement support and logical edge identity semantics | Keep support membership, logical edge IDs, physical slots, holes, masks, and active-support generations distinct. |
| `CE-CCP1-C03-007` | Implement orientation and transpose semantics | Model forward, transpose/backward, and orientation-specific output axes as semantic operation choices over shared logical edges, not pointer reinterpretations. |
| `CE-CCP1-C03-008` | Implement numerical tuple semantics | Carry storage, dense input, compute, accumulation, output, nonfinite, precision, and approximation contracts before candidate selection. |
| `CE-CCP1-C03-009` | Implement operation-kind resolution | Resolve relation apply, transpose, support contraction, segment statistics, normalization, edge map/gate, sparse update, bundle, chain, moments, hierarchy, exchange, gradient, and publication kinds. |
| `CE-CCP1-C03-010` | Implement output/update effect semantics | Distinguish assign, add, subtract, multiply, maximum, shared-destination accumulation, partial outputs, canonicalization, and epilogues with explicit alias legality. |
| `CE-CCP1-C03-011` | Implement structure, value, and support generation typing | Track structure epoch, value generation, active-support generation, order generation, and publication state separately in Sema. |
| `CE-CCP1-C03-012` | Implement persistence and identity typing | Distinguish inferred identity, declared persistent identity, user-forced identity, cloned identity, and ephemeral compiler handles. |
| `CE-CCP1-C03-013` | Integrate C++ templates and concepts with biological constraints | Expose concepts/traits for semantic categories while ensuring final Cellerator operation selection sees instantiated C++ numeric and layout types. |
| `CE-CCP1-C03-014` | Implement explicit low-level casts and escape hatches | Provide checked, trusted, and unsafe conversions between ordinary C++ views and compiler-semantic objects, with explicit effect/identity contracts. |
| `CE-CCP1-C03-015` | Create semantic validation and explanation APIs | Return structured compatibility results explaining domains, orders, generations, numerical policies, and operation resolution. |
| `CE-CCP1-C03-016` | Freeze biological Sema conformance | Lower all current operation-problem and relation-algebra fixtures through frontend Sema and prove no semantic information needed by planning is lost. |

### C04: fields, effects, profiles, and control

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-C04-001` | Define execution-field semantic ownership | Treat a field as one explicit semantic optimization and planning region with captured values, observable boundaries, profile environment, effect summary, and stable inspectable identity. |
| `CE-CCP1-C04-002` | Resolve and implement nested-field semantics | Use nested fields as separately nameable subregions whose constraints overlay inherited facts; the inner field is a planning subproblem and an optimization boundary unless explicitly inlined by the programmer. |
| `CE-CCP1-C04-003` | Implement statement ordering and observable effects | Permit reordering/fusion only when data dependencies, C++ observable effects, generations, numerical contracts, and field constraints prove equivalence. |
| `CE-CCP1-C04-004` | Implement opaque native-call barriers | Convert uncontracted C/C++ calls into explicit field barriers that conservatively invalidate affected profile/generation state and stop cross-call planning. |
| `CE-CCP1-C04-005` | Implement native effect contracts | Attach reads, writes, topology/order/support/value mutation, purity, determinism, alias, publication, and target behavior to resolved C++ functions. |
| `CE-CCP1-C04-006` | Implement automatic lifetime and generation transfer | Propagate structure/value/support/order generations through known operations, loops, native contracts, and field exits; materialize explicit transitions in semantic state. |
| `CE-CCP1-C04-007` | Implement persistence and reuse facts | Represent stable topology, mutable values, slowly evolving support, stable order, reuse horizon, recurrence, loop invariance, epoch boundary, and invalidation as source-level planning facts. |
| `CE-CCP1-C04-008` | Implement named representative-profile binding | Bind one or more compile-supplied profile states to fields and operations, support explicit state selection/aliasing, and keep data paths outside language semantics. |
| `CE-CCP1-C04-009` | Implement expected data-state transformation hints | Allow operations/native calls to state or select expected post-transform profile states, support inferred transfer functions, and warn when costly widening is required. |
| `CE-CCP1-C04-010` | Implement conditional profile alternatives and joins | Carry bounded branch-conditioned alternatives and explicit joins without generating uncontrolled decision-tree specialization. |
| `CE-CCP1-C04-011` | Implement planning facts and preferences | Apply non-binding facts/objectives for reuse, memory, latency, throughput, compilation budget, target preference, graph capture, and canonical output without changing mathematical meaning. |
| `CE-CCP1-C04-012` | Implement hard semantic and execution constraints | Restrict legal plans for determinism, numerical tolerance, exactness, memory bounds, target capabilities, candidate families, order, and synchronization. |
| `CE-CCP1-C04-013` | Implement custom candidate and forced realization controls | Bind source declarations that offer a custom candidate to the planner or force an exact candidate/decomposition/realization, including explicit unsafe modes. |
| `CE-CCP1-C04-014` | Implement missing-profile failure policy | Fail activated biological compilation when no representative semantic profile is bound, while allowing pure C++ fallthrough, CEIR-only structural work, and explicitly selected generic reference profiles. |
| `CE-CCP1-C04-015` | Implement field-level reflection identity | Assign stable field handles accessible to later reflection, cross-TU export, provenance, and celleratord without embedding runtime metadata. |
| `CE-CCP1-C04-016` | Deliver the first profile-required semantic field slice | Compile a typed relation field through source, C++ resolution, biological Sema, profile binding, effect/lifetime analysis, and a semantic operation problem without selecting physical execution yet. |
