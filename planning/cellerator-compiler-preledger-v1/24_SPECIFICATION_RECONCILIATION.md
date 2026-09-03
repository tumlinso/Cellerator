# Specification reconciliation

The language and IR documents guide implementation, but the implementation plan resolves several areas that require explicit reconciliation.

## Resolved clarifications

### Compiler architecture

Any text implying that Cellerator must be a Clang fork is superseded. The frontend uses an independent activated grammar and shadow-C++/upstream-Clang semantic bridge behind a versioned adapter.

### Nested fields

Nested fields are accepted as separately nameable planning subproblems with inherited facts, overlaid constraints, and an optimization boundary. Explicit semantic inlining may dissolve the boundary.

### Missing profiles

Activated biological semantic compilation without any bound representative profile is an error. Structural tooling and pure C++ fallthrough are not falsely treated as data-aware compilation. Generic reference profiles require explicit selection.

### IR validation

More paternalistic mandatory verification wording is reconciled to:

- structurally uninterpretable input fails;
- verified and checked modes enforce requested validation;
- trusted, unsafe, and unchecked modes may continue when the IR/backend can represent the request;
- the compiler explains risk and invalidated guarantees.

### Same-compilation transforms

Part One uses compiler preludes plus early host compilation/loading and bounded meta-generations. This is a supported same-compilation transform model without committing to the full deferred JIT architecture.

### Planning IR

Planning IR primarily represents the search space and may also carry selected/forced state in the same representation. A separate opaque winner-only IR is rejected.

### CellShard ownership

Old JBC claims that CellShard owns discovery, grammar, basis, global graph, or portable schedule compilation are superseded. Concrete storage/runtime remains downstream.

### Public IR levels

Semantic, Planning, and Realization IR remain the public family. Existing finer-grained artifacts become facets, imports/exports, or resumption points rather than eight public languages.

### Direct PTX

Typed direct PTX belongs as a target-specific Realization IR extension/backend path. Raw payloads require explicit inputs, outputs, clobbers/effects, target requirements, and trust mode.

### libCellerator

Compiler and runtime/execution APIs are both public constituencies. The SDK is componentized and also provides a convenience umbrella.

## Specification reconciliation Todos

The actual documents are revised only during implementation and final integration, especially `CE-CCP1-J03-003` through `CE-CCP1-J03-005`. No source specification is silently rewritten by this planning package.

## Open issue deliberately retained

The word `atom` remains in force. A rename is not authorized by this plan.
