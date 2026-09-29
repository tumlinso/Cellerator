# Source map: one path through Cellerator

Read a semantic contract, its realization and one consumer before exploring the whole tree.

| Question | Start here | What to look for |
|---|---|---|
| Where is the host compiler configured? | [compiler target graph](../../{{P_COMPILER}}) | Compiler components versus optional CUDA/runtime routes. |
| What is the library-facing compiler surface? | [compiler API](../../{{P_API}}) | Parsing/semantics, profiles, planning, realization and diagnostics; coverage differs by path. |
| Where does biological support become geometry? | [geometry](../../{{P_GEOMETRY}}) | Reusable structure, order and exact contribution ownership. |
| Where are prepared operations executed? | [execution](../../{{P_EXECUTION}}) | Binding/lifetime, values, candidate realization and program execution. |
| Can source and native semantics share an executor? | [bounded relation-update example](../../{{P_UPDATE_EXAMPLE}}) | Same operation/gradient/update semantics through the recorded routes. |
| Where is the framework boundary? | [CelleraTorch](../../{{P_ADAPTER}}) | Explicit supported views/gradients; not a duplicate numerical owner. |
| Where do old interfaces live? | [compatibility material](../../{{P_LEGACY}}) | Useful evidence and remaining callers, not the default architecture. |

For the detailed design and migration record, continue to [Architecture](../architecture.qmd), [Current Implementation history](../current_implementation.qmd), and [Migration Roadmap](../migration_roadmap.qmd). The [root build](../../{{P_CORE_CMAKE}}) is authoritative for actual targets. These are the current source entry paths. No source-path move was justified in this outcome; the compiler, geometry, and execution groupings remain in place.
