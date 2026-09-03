# Semantic IR implementation plan

## Public CEIR family

The implementation retains three public levels:

1. Semantic IR: biological program meaning, effects, generations, field boundaries, control flow, and profile references.
2. Planning IR: unresolved and selected alternatives, discovery, exact coverage, atoms, decomposition, candidates, costs, and decisions.
3. Realization IR: committed physical cover, extents, projections, packing, stages, readiness, bindings, targets, and native operations.

Current internal records such as CSG1, CPE2, atom evidence, semantic atoms, target cover, packed operands, and executable recipes become import/export or resumption facets. They do not each become a separate public language.

## Common CEIR object model

All levels share:

- contexts, arenas, immutable snapshots, and mutable builders;
- types, attributes, operations, regions, blocks, values, and use-def chains;
- semantic, persistent, artifact, compiler-local, and provenance identities;
- extension namespaces and opaque preservation;
- verified, checked, trusted, unsafe, and unchecked modes;
- canonical text and sectioned binary artifacts;
- removable provenance sidecars;
- direct standalone compilation checkpoints.

## Semantic IR scope

Semantic IR owns modules, translation units, functions, named/anonymous fields, domains, axes, states, relations, operation families, native calls, control flow, generations, profile references, and inlining.

Equivalent source spellings may canonicalize to the same meaning. Original syntax remains available through provenance, not by distorting the semantic form.

Typed numeric policy remains early and first-class.

## Textual round trip

Required:

- source -> semantically equivalent IR;
- CEIR text -> IR object -> identical canonical CEIR text;
- binary CEIR -> equivalent IR and canonical text.

Reconstructed original Cellerator source is optional and is not allowed to constrain canonicalization.

## Workstream task catalog

### D01: common CEIR

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-D01-001` | Freeze CEIR common lexical conventions | Define identifiers, stable/persistent identities, SSA values, types, attributes, regions, comments, profile references, native payloads, extension namespaces, and abstraction-level markers in a compact Cellerator-native textual grammar. |
| `CE-CCP1-D01-002` | Implement arena and ownership model | Provide mutable builders, immutable snapshots, copy-on-write editing, explicit contexts, and allocation-free traversal views for all public IR levels. |
| `CE-CCP1-D01-003` | Implement common type and attribute interning | Intern structural types and attributes by canonical content while allowing opaque extension payloads and user-controlled identity assertions. |
| `CE-CCP1-D01-004` | Implement regions, blocks, values, and use-def chains | Provide typed values, regions, blocks, explicit control edges, use lists, mutation APIs, and stable handles suitable for source reflection and pass rewriting. |
| `CE-CCP1-D01-005` | Implement common operation and extension records | Represent operation name, dialect/namespace, operands, results, regions, attributes, effects, source provenance, validation mode, and unknown opaque payloads. |
| `CE-CCP1-D01-006` | Implement persistent and transient identity layers | Separate semantic/persistent IDs, artifact/content IDs, compiler-local handles, and optional provenance IDs; allow identities to be stripped from hot lowerings when not required. |
| `CE-CCP1-D01-007` | Implement removable provenance sidecars | Store source spans, transform lineage, profile evidence, planner decisions, and backend mappings in cold sidecars or removable artifact sections. |
| `CE-CCP1-D01-008` | Implement the CEIR text lexer and parser framework | Build reusable error-recovering parsing with dialect dispatch, source includes/imports, inline block support, and precise byte ranges. |
| `CE-CCP1-D01-009` | Implement deterministic canonical printing | Print stable ordering, normalized whitespace, explicit versions, and lossless opaque extension payloads while offering a readable noncanonical pretty mode. |
| `CE-CCP1-D01-010` | Implement sectioned binary CEIR serialization | Design a versioned, checksummed, pointer-free, memory-mappable container for large IR modules and cold metadata, with textual form remaining authoritative for hand programming. |
| `CE-CCP1-D01-011` | Implement standalone CEIR compiler input detection | Allow cellerator to accept semantic, planning, or realization `. |
| `CE-CCP1-D01-012` | Implement validation-mode plumbing | Carry verified, checked, trusted, unsafe, and unchecked modes through parser, builder, pass, serializer, and backend handoff without conflating structural parseability with semantic trust. |
| `CE-CCP1-D01-013` | Expose the public CEIR C++ API | Publish contexts, builders, readers, writers, iterators, diagnostics, extension hooks, and ownership rules without exposing Clang/LLVM implementation objects. |
| `CE-CCP1-D01-014` | Freeze CEIR common round-trip and artifact compatibility | Demonstrate text and binary round-trip, unknown extension preservation, standalone input, and source-inline parsing through one common interface. |

### D02: Semantic IR

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-D02-001` | Freeze Semantic IR module and symbol scopes | Define program, module, translation-unit, function, named field, anonymous field, and imported semantic symbol ownership with explicit cross-TU export authorization. |
| `CE-CCP1-D02-002` | Implement domain and axis IR types | Represent domain/tag, extent knowledge, order, geometry, partition, local/global identity, and recovery contracts as distinct typed entities. |
| `CE-CCP1-D02-003` | Implement state and value-plane IR types | Represent axis, width, storage/compute/accumulation/output types, order, generation, mutability, address/residency intent, and alias class. |
| `CE-CCP1-D02-004` | Implement relation IR types | Represent source/destination axes, structure identity/epoch, logical edge identity/order/count, support, value plane/generation, active-support generation, orientation, and mutation policy. |
| `CE-CCP1-D02-005` | Implement execution-field operations and regions | Represent field boundaries, captures, results, profile environment, facts, constraints, observable effects, and explicit nested-field semantics. |
| `CE-CCP1-D02-006` | Implement relation-apply and transpose operations | Model apply and transpose as typed semantic operations with explicit source/destination/result effects and numerical policy, independent of sparse/dense format. |
| `CE-CCP1-D02-007` | Implement contraction, segment, and normalization operations | Represent support-restricted contractions, segmented statistics/reductions, and normalization with segment identities, neutral elements, determinism, and output effects. |
| `CE-CCP1-D02-008` | Implement edge map, gate, support-mask, and sparse-update operations | Represent logical-edge iteration, projection-independent gates, dynamic support generations, and axis-indexed updates with alias and order semantics. |
| `CE-CCP1-D02-009` | Implement bundle, chain, moments, hierarchy, and exchange operations | Represent grouped relations, multi-hop chains, paired moments, incidence pool/broadcast, and typed exchange as semantic graphs with explicit intermediate axes and effects. |
| `CE-CCP1-D02-010` | Implement gradient and publication operations | Represent forward/transpose/value-gradient closure, generation publication, canonicalization, and caller-owned update-policy boundaries without importing model/loss semantics. |
| `CE-CCP1-D02-011` | Implement native and opaque C++ call operations | Represent resolved symbol, typed operands/results, effect contract or conservative barrier, exceptions, determinism, and provenance. |
| `CE-CCP1-D02-012` | Implement control flow and loop semantics | Use structured regions plus explicit branches/loops where possible, carry bounded profile alternatives, and preserve opaque C++ control when semantic extraction is unavailable. |
| `CE-CCP1-D02-013` | Implement generation and epoch transition operations | Make structure/value/support/order invalidation, publication, cloning, assertion, and epoch boundary explicit operations rather than hidden metadata changes. |
| `CE-CCP1-D02-014` | Implement Semantic IR inlining and composition | Inline functions, operations, and fields with capture substitution, identity policy, profile-state substitution, generation repair, and provenance retention. |
| `CE-CCP1-D02-015` | Implement semantic canonicalization and equivalence | Normalize source spellings to meaning while preserving relevant type, effect, identity, numerical, and field boundaries. |
| `CE-CCP1-D02-016` | Deliver source-to-Semantic-IR vertical slice | Lower a profile-bound relation field and a mixed multi-operation field from `. |
