# Source map: one path through Cellerator

Read a semantic contract, its realization and one consumer before exploring the whole tree.

| Question | Start here | What to look for |
|---|---|---|
| Where is the host compiler configured? | [compiler target graph](../../src/compiler/CMakeLists.txt) | Compiler components versus optional CUDA/runtime routes. |
| What is the library-facing compiler surface? | [compiler API](../../include/Cellerator/compiler/api/compiler.hpp) | Parsing/semantics, profiles, planning, realization and diagnostics; coverage differs by path. |
| Where does biological support become geometry? | [geometry](../../src/geometry) | Reusable structure, order and exact contribution ownership. |
| Where are prepared operations executed? | [execution](../../src/execution) | Binding/lifetime, values, candidate realization and program execution. |
| Can source and native semantics share an executor? | [bounded relation-update example](../../examples/relation_update_spine_v1/regulatory_learning.cc) | Same operation/gradient/update semantics through the recorded routes. |
| Where is the framework boundary? | [CelleraTorch](../../components/CelleraTorch) | Explicit supported views/gradients; not a duplicate numerical owner. |
| Where do old interfaces live? | [compatibility material](../../compat) | Useful evidence and remaining callers, not the default architecture. |

For the detailed design and migration record, continue to [Architecture](../architecture.qmd), [Current Implementation history](../current_implementation.qmd), and [Migration Roadmap](../migration_roadmap.qmd). The [root build](../../CMakeLists.txt) is authoritative for actual targets. These are the current source entry paths. No source-path move was justified in this outcome; the compiler, geometry, and execution groupings remain in place.
