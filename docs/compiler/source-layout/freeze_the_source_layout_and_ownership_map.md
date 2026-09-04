# Source layout and ownership map freeze receipt

Task: `CE-CCP1-A04-010`

Published contract: `SOURCE_LAYOUT_V1.md`

Consumed contract: `CE-CCP1-I03-COMPILER-OWNERSHIP` version 1, frozen hash
`3478e9787fbee8e66f1c12dba0d69641d01605ef2316420f7d74cac02421b1d0`.

## Collision analysis

| Candidate path | Current state | Result |
| --- | --- | --- |
| `include/Cellerator/compiler/` | absent | free; does not collide with `include/Cellerator/geometry/compiler/` |
| `src/compiler/` | absent | free; geometry compiler remains under `src/geometry/compiler/` |
| `tools/cellerator/`, `tools/celleratord/` | absent | free; current `tools/` files remain |
| `stdlib/`, `profiles/reference/`, `bench/compiler/` | absent | free; new owned resource/benchmark roots |
| `tests/compiler/` | contains A04 source-layout gates | compatible reserved compiler test root |
| `include/Cellerator/compiler.hh`, `include/Cellerator/runtime.hh` | absent | reserved central umbrellas; current `compute/runtime.hh` is distinct |
| `include/Cellerator/Cellerator.hh` | exists | preserved and integration-only |

No planned destination aliases an existing runtime, geometry, compute, planner,
or CellShard ownership root.

## Concurrent-write analysis

Leaf writes are disjoint by declared Project Control scope. Work within one
component lane is serial; cross-lane contracts cross only frozen interfaces or
queued integration artifacts. Singleton central files have no leaf writer and
exactly one active integration owner. Provider additions use stable-ID fragments
in provider-owned subtrees, so concurrent providers do not edit a common list.

The A04 lane writes only `docs/compiler/source-layout/` and its focused
`tests/compiler/a04/` files. It has not edited central CMake, package, umbrella,
registry, manifest, or gitlink paths.

## Dependency checks

- `CE-CCP1-A04-001` through `CE-CCP1-A04-009` are the source-linked inputs to
  `SOURCE_LAYOUT_V1.md`.
- `CE-CCP1-CP-A03` was reached before this freeze.
- `CE-CCP1-I03-COMPILER-OWNERSHIP` version 1 is the ownership source; this map
  does not broaden or reinterpret it.
- `CE-CCP1-I04-SOURCE-LAYOUT` version 1 is consumed by the compiler build-graph
  owner only after publication through Project Control.

## Validation evidence

`tests/compiler/a04/freeze_the_source_layout_and_ownership_map_test.cc` checks
all path-map categories, current collision facts, uniqueness of central locks,
lane/dependency rules, frozen I03 identity, source-detail links, and the Part One
boundary.
