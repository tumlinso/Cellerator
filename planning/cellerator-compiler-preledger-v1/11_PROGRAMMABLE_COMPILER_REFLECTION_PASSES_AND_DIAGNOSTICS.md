# Programmable compiler, reflection, passes, and diagnostics

## Reflection

Normal `.cell` source can acquire typed compiler handles for fields, operations, relations, profile states, Planning alternatives, selected realization, costs, and provenance at the phase where they exist.

Reflection participates in templates and compile-time construction.

## Inline IR

Semantic, Planning, and Realization IR may be embedded in source with typed captures/results. Inline blocks can inspect, insert, replace, wrap, or force compiler structure without abandoning the source language.

## Pass pipeline

Users can insert or replace meaningful stages across:

- semantic canonicalization;
- profile propagation;
- discovery and certification;
- decomposition and candidate enumeration;
- cost modeling and selection;
- realization, projection, packing, and stages;
- backend emission.

Extensions may define custom operations, types, attributes, profile dimensions, atoms, costs, realization nodes, and backend operations. Unknown extensions are preserved where continuation is technically possible.

## Same-compilation transforms

A compiler-prelude region is parsed first. Transform code is compiled for the host into a temporary cached extension artifact, loaded, and applied to later phases of the same compilation. Meta-generation is bounded and diagnosed. This is a focused staged compiler mechanism, not the deferred general JIT product.

## Validation philosophy

- structural impossibility hard-fails;
- verified/checked modes diagnose and enforce requested contracts;
- trusted/unsafe/unchecked modes continue when representation/backend can proceed;
- warnings explain consequences rather than pretending the compiler is authority over expert code.

## Provenance

Source-to-native provenance is cold metadata. It can be embedded, sidecar, separately debugged, or stripped without affecting hot execution.

## Workstream task catalog

### G01: reflection and inline IR

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-G01-001` | Freeze the compile-time IR handle model | Define typed handles for source declarations, fields, operations, relations, profile states, Planning alternatives, selected realizations, and provenance with explicit availability phase and lifetime. |
| `CE-CCP1-G01-002` | Implement reflection of current and named execution fields | Allow source code to acquire the current field or a named/exported field as Semantic IR during the valid compiler phase. |
| `CE-CCP1-G01-003` | Implement reflection of operations and relations | Expose normalized operation kind, typed operands/results, relation/domain/order/generation semantics, effects, and source provenance. |
| `CE-CCP1-G01-004` | Implement reflection of profile environments | Expose named states, attached evidence summaries, confidence, mutation expectations, joins, unknown dimensions, and selected state at a source location. |
| `CE-CCP1-G01-005` | Implement reflection of Planning IR search spaces | Expose alternatives, exact coverage, atoms, decompositions, candidates, costs, rejection reasons, and selection state after planning becomes available. |
| `CE-CCP1-G01-006` | Implement reflection of Realization IR | Expose selected cover, extents, projections, packing, stage graph, resources, generations, and native/backend fragments after realization. |
| `CE-CCP1-G01-007` | Integrate reflection with C++ templates and constant evaluation | Allow compile-time branching, concepts, and type/value construction over stable reflection queries without requiring standardized C++ reflection. |
| `CE-CCP1-G01-008` | Implement compile-time CEIR builders | Provide source-accessible builders for typed CEIR nodes, attributes, regions, alternatives, candidates, and native fragments with deterministic allocation and diagnostics. |
| `CE-CCP1-G01-009` | Implement inline Semantic IR blocks | Splice typed source captures/results into Semantic IR, validate domain/effect/generation contracts, and retain surrounding field/profile context. |
| `CE-CCP1-G01-010` | Implement inline Planning IR blocks | Allow additions, removals, cost changes, decompositions, candidate offers, force directives, and planner replacement fragments at explicit planning points. |
| `CE-CCP1-G01-011` | Implement inline Realization IR blocks | Allow explicit projections, packs, stages, target operations, and native fragments with typed bindings and chosen validation mode. |
| `CE-CCP1-G01-012` | Implement typed capture and result binding | Bind source variables, C++ expressions, CEIR values, profile states, runtime slots, and generated symbols without hidden copies or ambiguous lifetimes. |
| `CE-CCP1-G01-013` | Implement operation replacement and IR splicing | Support before/after/replace/wrap/inlining splices with explicit dominance, effects, identities, profiles, and result substitution. |
| `CE-CCP1-G01-014` | Define identity and generation behavior under inline IR | Preserve identity only when explicitly semantically valid; otherwise derive new IDs, update generations, and retain lineage in cold provenance. |
| `CE-CCP1-G01-015` | Expose reflection and inline-IR diagnostics | Report phase availability, type mismatch, stale handles, capture effects, unknown extensions, validation mode, and resulting compiler invalidations. |
| `CE-CCP1-G01-016` | Deliver the first source-defined inline rewrite | Compile a `. |

### G02: passes, extensions, self-transforms

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-G02-001` | Freeze the pass pipeline stage taxonomy | Name stable interception points before/after source canonicalization, profile propagation, discovery, certification, decomposition, candidate enumeration, cost modeling, selection, realization, packing, stage construction, and backend emission. |
| `CE-CCP1-G02-002` | Implement pass manager and analysis invalidation | Provide ordered pipelines, nested scopes, required/preserved analyses, explicit invalidation, cancellation, timing, diagnostics, and deterministic replay. |
| `CE-CCP1-G02-003` | Implement pipeline configuration syntax and APIs | Accept source directives, inline Planning IR, command-line pipeline text, C++ API builders, and profile-specific pipelines with one normalized representation. |
| `CE-CCP1-G02-004` | Implement custom semantic pass API | Expose Semantic IR read/write access, profiles, source mappings, diagnostics, and analysis cache while allowing semantics-changing transformations under declared validation mode. |
| `CE-CCP1-G02-005` | Implement custom planning pass API | Expose search spaces, evidence, atoms, decomposition, candidates, costs, selection, and planner services for additions or full replacement. |
| `CE-CCP1-G02-006` | Implement custom realization pass API | Expose physical covers, projections, packs, stages, bindings, target operations, and native fragments before backend emission. |
| `CE-CCP1-G02-007` | Implement complete built-in stage replacement | Allow profile propagation, discovery, certification, decomposition, candidate enumeration, cost model, planner, realization, and backend lowering to be replaced rather than merely decorated. |
| `CE-CCP1-G02-008` | Define extensible operation, type, and attribute registration | Register custom namespaces, text syntax, types, attributes, effects, reflection, state transfer, verification, cost, and lowering protocols independently so extensions implement only what they need. |
| `CE-CCP1-G02-009` | Implement unknown extension preservation and forwarding | Parse, print, serialize, clone, inline, import, and pass through unknown extension nodes at levels where opaque continuation is technically possible. |
| `CE-CCP1-G02-010` | Implement extension capability negotiation | Record required protocols and backend capabilities; allow inspect-only, preserve-only, external-lowered, and fully understood modes. |
| `CE-CCP1-G02-011` | Implement compiler prelude regions | Define source regions parsed and semantically resolved before the main compilation graph for transform declarations, extension schemas, and pipeline configuration. |
| `CE-CCP1-G02-012` | Compile same-translation-unit transforms in an early host stage | Use the selected host toolchain to compile prelude transform code into a temporary compiler extension artifact, keyed by content/toolchain/API identity, without requiring the deferred general JIT. |
| `CE-CCP1-G02-013` | Implement bounded meta-generation and staging | Allow a transform generation to affect later phases of the same compilation, cap recursive self-generation by explicit policy, and provide deterministic cycle/phase diagnostics. |
| `CE-CCP1-G02-014` | Cache compiled transforms safely | Cache prelude extension artifacts by source, compiler API, ABI, toolchain, target-host, dependencies, and trust policy; invalidate precisely and support keep-temps. |
| `CE-CCP1-G02-015` | Implement transform sandbox policy as opt-in, not authority | Offer subprocess isolation, time/memory limits, and verified modes while allowing trusted in-process transforms and unsafe continuation for experts. |
| `CE-CCP1-G02-016` | Integrate extension/pass provenance | Record extension identity, binary/source hash, pipeline location, inputs, outputs, invalidated analyses, diagnostics, and trust mode in cold provenance. |
| `CE-CCP1-G02-017` | Deliver same-compilation self-transformation | Compile a translation unit that defines a prelude pass, reflects a field, rewrites Semantic or Planning IR in the same compilation, and emits a correct ordinary object. |
| `CE-CCP1-G02-018` | Freeze the open compiler-extension surface | Publish pass, pipeline, extension, staging, cache, trust, and provenance contracts for libCellerator and celleratord. |

### G03: validation, diagnostics, provenance

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-G03-001` | Freeze validation-mode semantics | Define verified, checked, trusted, unsafe, and unchecked behavior separately for parsing, semantic invariants, exact coverage, numerical claims, resources, native bindings, and backend support. |
| `CE-CCP1-G03-002` | Implement structural impossibility checks | Fail only for malformed graphs, missing required operands, impossible references, uninterpretable text, or backend states that cannot be represented/continued. |
| `CE-CCP1-G03-003` | Implement advisory semantic validators | Detect domain/order/generation/effect/numerical/identity inconsistencies and permit explicit trusted/unsafe continuation where the IR remains representable. |
| `CE-CCP1-G03-004` | Implement exact-coverage and ownership diagnostics | Explain omissions, duplicates, wrong roles, incompatible partial algebra, halo/contributor confusion, and canonical recovery failure with member-level evidence. |
| `CE-CCP1-G03-005` | Implement target/native diagnostics | Explain unsupported instructions, capability ranges, clobbers, alignment, address spaces, collectives, ABI, graph capture, and fallback availability. |
| `CE-CCP1-G03-006` | Build the provenance graph model | Trace source, AST, Semantic IR, profile evidence, passes, Planning alternatives, selection, Realization stages, generated source, backend objects, and native symbols through cold IDs and edges. |
| `CE-CCP1-G03-007` | Implement source-to-native maps and removable sections | Emit sidecars or debug/object sections for provenance without adding hot runtime fields; support strip and separate-debug workflows. |
| `CE-CCP1-G03-008` | Implement 'what changed' and staleness explanations | Explain structure/value/support/order/profile generations changed by each statement or pass, which artifacts became stale, and earliest lowering resumption point. |
| `CE-CCP1-G03-009` | Implement planning barrier explanations | Identify opaque C++ calls, field boundaries, unknown extensions, effects, alias uncertainty, profile widening, and hard constraints that stop fusion or movement. |
| `CE-CCP1-G03-010` | Implement candidate decision reports | Explain legal candidates, rejected reasons, complete costs, evidence freshness/uncertainty, user edits, selected source, forced choices, and fallback. |
| `CE-CCP1-G03-011` | Implement optimization remarks and missed-opportunity diagnostics | Provide opt-in remarks for persistence assumptions, missing profile hints, expensive canonicalization, avoidable packing, unshared orders, and uncertain branches. |
| `CE-CCP1-G03-012` | Implement deterministic reproducer bundles | Capture source subset, profiles, CEIR checkpoints, toolchain manifest, pipeline, extensions, diagnostics, and commands needed to reproduce compiler failures without runtime dataset payloads. |
| `CE-CCP1-G03-013` | Implement crash and timeout diagnostics | Attribute failures to compiler phase/pass/backend, preserve temporary artifacts, and isolate optional custom passes when configured. |
| `CE-CCP1-G03-014` | Expose structured diagnostic and query APIs | Publish C/C++ callbacks, JSON/LSP records, stable IDs, related-information links, and cancellation-safe streaming for cellerator, libCellerator, and celleratord. |
| `CE-CCP1-G03-015` | Benchmark diagnostics and provenance overhead | Measure disabled, minimal, and full provenance compile-time/RSS/object-size overhead; keep hot runtime overhead zero. |
| `CE-CCP1-G03-016` | Freeze explainability and unsafe-control behavior | Demonstrate verified failure, trusted continuation, unsafe native lowering, full candidate explanation, source-to-native trace, and provenance stripping. |
