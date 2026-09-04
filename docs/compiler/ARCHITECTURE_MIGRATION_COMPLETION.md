# Part One architecture and migration completion

The final compiler tree is split by source/frontends, semantic analysis,
profiles, CEIR levels, planning, realization, reflection, passes, LTO, APIs,
SDK, tooling, profiles, and standard-library resources. Stable contracts live
under `include/Cellerator/compiler`; implementations live under `src/compiler`.

Cellerator owns compiler meaning and selection. CellShard retains concrete
storage, persistence, placement, transport, materialization, and runtime
binding. Preserved JBC history is pinned to the migration manifests and the
embedded source revision; every family has a Cellerator destination, retained
compatibility disposition, or explicit non-promotion record.

Prior planning charters are historical evidence. Frozen Project Control
interfaces govern the implemented ABI. Host-only, optional NVIDIA, SDK,
standard-library, and tooling modes share the compiler libraries. General JIT
and deep CellShard runtime evolution remain versioned Part Two seams.
