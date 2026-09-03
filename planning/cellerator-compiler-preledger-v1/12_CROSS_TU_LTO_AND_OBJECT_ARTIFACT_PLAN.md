# Cross-TU, LTO, and object-artifact plan

## Ordinary-object coexistence

Cellerator emits ordinary objects. Compiler metadata may be:

- embedded in non-loadable ELF/Mach-O/COFF sections;
- compressed when supported;
- retained in static archives;
- emitted as content-addressed sidecars when embedding is unavailable.

The object remains linkable by conventional tools.

## Explicit authorization

Cross-TU optimization is permitted only for exported/named Cellerator fields, explicit source policy, or driver LTO mode. Arbitrary plain C++ is not absorbed into a biological planning region.

## Program compilation

At link time, the Cellerator driver:

1. extracts CEIR/profile summaries from objects and archives;
2. resolves symbols, templates, fields, and profile identities;
3. imports full field IR on demand;
4. builds program-level Semantic/Planning IR;
5. performs authorized inlining, connected planning, order reuse, and shared realization;
6. regenerates affected CPU/CUDA/native objects;
7. invokes the selected conventional linker.

Incremental summaries permit thin-style reuse.

## Workstream task catalog

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-H01-001` | Freeze the CEIR companion object-artifact contract | Define versioned Semantic/Planning summary, profile references, field exports, provenance maps, toolchain identity, and content hashes embedded in or associated with ordinary objects. |
| `CE-CCP1-H01-002` | Implement ELF CEIR sections | Emit deterministic non-loadable sections, symbols/notes, compression policy, strip behavior, and extraction without affecting runtime ABI. |
| `CE-CCP1-H01-003` | Implement Mach-O and COFF strategies | Define platform sections or sidecar fallback with identical semantic content and deterministic naming where platform toolchains are available. |
| `CE-CCP1-H01-004` | Implement portable sidecar fallback | Emit content-addressed `. |
| `CE-CCP1-H01-005` | Assign cross-TU field and symbol identities | Resolve exported domains, relations, fields, profiles, passes, native symbols, and template instantiations using semantic fingerprints plus linkage/module identity. |
| `CE-CCP1-H01-006` | Implement template instantiation deduplication | Deduplicate equivalent C++/Cellerator template specializations while preserving distinct numeric/domain/profile semantics and backend variants. |
| `CE-CCP1-H01-007` | Implement profile environment merge | Merge named states and evidence references by stable identity/revision, diagnose conflicting biological semantics, and retain bounded alternatives. |
| `CE-CCP1-H01-008` | Implement cross-TU semantic imports | Import exported Semantic IR summaries or full field bodies on demand with source/provenance references and extension negotiation. |
| `CE-CCP1-H01-009` | Implement explicit program-planning authorization | Require exported/named fields, source policy, or driver LTO flags before crossing ordinary field/TU boundaries; never absorb arbitrary plain C++ globally. |
| `CE-CCP1-H01-010` | Implement object and archive CEIR extraction | Scan individual objects, static archives, shared-library metadata, and linker inputs without loading code; index fields/profiles by identity. |
| `CE-CCP1-H01-011` | Implement Cellerator link-driver mode | Intercept link invocations, build the program CEIR graph, run authorized Cellerator LTO, produce replacement/additional objects, then invoke the selected conventional linker. |
| `CE-CCP1-H01-012` | Implement program-level Semantic/Planning IR | Merge field graphs, cross-function calls, profile families, external effects, program constraints, and shared artifacts without changing unexported semantics. |
| `CE-CCP1-H01-013` | Implement cross-TU inlining and connected planning | Inline semantic fields/functions and optimize persistent orders, shared decompositions, candidates, and transitions where authorization and effects permit. |
| `CE-CCP1-H01-014` | Implement incremental and thin-summary LTO | Cache per-object summaries and full CEIR, invalidate by semantic/profile/toolchain/pass identity, and replan only affected program regions. |
| `CE-CCP1-H01-015` | Implement mixed-backend re-emission | Regenerate CPU/CUDA/native objects for changed program regions while retaining untouched conventional objects and valid backend artifacts. |
| `CE-CCP1-H01-016` | Freeze cross-TU and LTO vertical acceptance | Compile two `. |
