# celleratord plan

## Architecture

`celleratord` is an LSP server built on shared libCellerator snapshots.

It supplies ordinary C++ behavior through a compatible upstream clangd worker/proxy or reusable upstream components. Cellerator-specific parsing, AST, Sema, profiles, CEIR, planning, provenance, and source maps remain native services.

This avoids a permanent clangd fork while preserving C++ feature breadth.

## Baseline features

- diagnostics, completion, hover, navigation, rename, symbols, compile commands;
- source mapping between original `.cell` and shadow C++;
- incremental/cancellable frontend snapshots;
- host-only operation and explicit unavailable profile-dependent queries.

## Cellerator features

- domains, axes, relations, operation and field hover;
- profile state, generations, mutation, and staleness;
- Semantic/Planning/Realization IR at cursor;
- candidate, cost, rejection, and winner explanations;
- decomposition and stage-graph visualization;
- opaque-barrier and missed-optimization explanations;
- inline CEIR completion/validation;
- source-to-generated/native navigation.

Basic editing must remain responsive while deep planning queries run in cancellable background work.

## Workstream task catalog

### I01: core and C++ compatibility

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-I01-001` | Freeze the celleratord architecture | Define an LSP server over shared libCellerator snapshots with an upstream clangd worker/proxy or reusable upstream components for ordinary C++ features, without a permanent fork. |
| `CE-CCP1-I01-002` | Implement JSON-RPC and LSP transport | Support stdio, framing, request IDs, notifications, cancellation, progress, initialization, capabilities, graceful shutdown, and structured logging. |
| `CE-CCP1-I01-003` | Implement clangd worker discovery and lifecycle | Discover an overridable compatible clangd, launch/manage it when proxy mode is used, forward initialization/configuration, restart safely, and expose version diagnostics. |
| `CE-CCP1-I01-004` | Implement compile-command and project configuration | Consume compile_commands, cellerator toolchain/profile/backend flags, resource directories, response files, and per-file activation state. |
| `CE-CCP1-I01-005` | Implement virtual shadow-document mapping | Maintain transformed C++ documents for clangd while mapping positions, edits, diagnostics, symbols, and fix-its to original `. |
| `CE-CCP1-I01-006` | Implement incremental source and AST snapshots | Reuse source tokens, include state, shadow C++, C++ AST bridge, Cellerator AST, and Semantic IR for unchanged regions with dependency-aware invalidation. |
| `CE-CCP1-I01-007` | Implement document scheduling and cancellation | Prioritize active files, debounce edits, cancel stale parses/plans, bound background work, and isolate slow profile/planning requests from basic editing. |
| `CE-CCP1-I01-008` | Merge ordinary C++ and Cellerator diagnostics | Deduplicate/remap clangd and Cellerator diagnostics, preserve severities/fix-its/related information, and identify their originating phase. |
| `CE-CCP1-I01-009` | Forward completion, hover, navigation, and rename | Pass through ordinary C++ features with source mapping and merge Cellerator symbol results without changing normal C++ behavior. |
| `CE-CCP1-I01-010` | Implement workspace symbol and indexing foundations | Index C++ and Cellerator domains, relations, fields, profiles, passes, IR symbols, and cross-TU exports using compiler fingerprints. |
| `CE-CCP1-I01-011` | Implement host-only/no-profile editing behavior | Allow syntax, C++ semantics, AST, and structural CEIR features without CUDA or loaded profile, while marking profile-dependent analysis unavailable rather than pretending it exists. |
| `CE-CCP1-I01-012` | Expose reusable tooling snapshot APIs | Publish immutable query snapshots, source-position lookup, diagnostic streams, cancellation tokens, and background compilation hooks through libCellerator. |
| `CE-CCP1-I01-013` | Benchmark baseline editor latency | Measure startup, first diagnostics, incremental edit, completion, hover, navigation, memory, and clangd-proxy overhead on plain C++ and `. |
| `CE-CCP1-I01-014` | Deliver celleratord C++-parity milestone | Build bin/celleratord, open mixed ordinary C++/Cellerator workspaces, provide normal C++ diagnostics/navigation/completion plus Cellerator syntax diagnostics, and run without CUDA. |

### I02: Cellerator semantic tooling

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-I02-001` | Implement Cellerator syntax completion | Complete semantic declarations, field constructs, relations, operation families, effects, persistence/profile controls, reflection, passes, inline CEIR, and native blocks from parser context. |
| `CE-CCP1-I02-002` | Implement biological type and relation hover | Show domains/tags, axes/orders, relation endpoints, support, orientation, numerical tuple, mutability, identities, and current generations with source links. |
| `CE-CCP1-I02-003` | Implement field ownership and effect views | Show enclosing field, nested boundary, captured values, native barriers, reads/writes, observable effects, profile environment, and optimization visibility. |
| `CE-CCP1-I02-004` | Implement profile-state-at-cursor | Show selected/named states, inferred evidence, confidence, alternatives, joined/unknown dimensions, expected support/value/mutation state, and missing hints. |
| `CE-CCP1-I02-005` | Implement generation and staleness queries | Answer what structure/value/support/order generations are live, what changes after a statement, and which cached/planned artifacts become stale. |
| `CE-CCP1-I02-006` | Implement Semantic IR at cursor | Render or open the normalized field/operation Semantic IR, source mapping, effects, profiles, and extensions with navigation back to source. |
| `CE-CCP1-I02-007` | Implement Planning IR and candidate views | Show planning problem, exact cover, atom proposals/certification, decompositions, candidates, resources, costs, evidence, rejected reasons, and selected/forced state. |
| `CE-CCP1-I02-008` | Implement realization and decomposition visualization | Show selected atoms/extents, projection/packing, orders, partial trees, stage graph, dependencies, workspace, target, and graph-capture/readiness behavior. |
| `CE-CCP1-I02-009` | Implement 'why did this candidate win?' | Explain complete cost, evidence freshness/uncertainty, constraints, transition costs, reuse, alternatives, dominance, user edits, and fallback. |
| `CE-CCP1-I02-010` | Implement opaque-barrier and missed-optimization actions | Explain uncontracted native calls, alias/effect uncertainty, field boundaries, profile widening, missing persistence facts, and canonicalization costs; offer safe source fix-its where available. |
| `CE-CCP1-I02-011` | Implement source-to-native navigation | Navigate source operation to Semantic/Planning/Realization nodes, generated C++/CUDA/PTX, native symbols/resource reports, and reverse-map native diagnostics. |
| `CE-CCP1-I02-012` | Implement inline CEIR diagnostics and completion | Provide level-aware operations/types/attributes, captures, profiles, candidates, target instructions, extension namespaces, validation modes, and structural diagnostics. |
| `CE-CCP1-I02-013` | Benchmark advanced semantic queries | Measure latency/RSS/cancellation for profile propagation, candidate explanation, IR rendering, decomposition graph, and native navigation with cached and cold states. |
| `CE-CCP1-I02-014` | Freeze Cellerator-aware celleratord acceptance | Demonstrate all agreed baseline queries over an installed multi-profile relation project, including IR at cursor, candidate costs, mutation staleness, decomposition, and source-to-native navigation. |
