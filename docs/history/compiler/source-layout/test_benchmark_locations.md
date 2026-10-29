# Compiler test and benchmark location freeze

Status: frozen for the Part One source-layout interface candidate.

Task: `CE-CCP1-A04-005`

## Validation ownership

Each planned compiler validation family has one owner path:

| Family | Test owner | Benchmark owner |
| --- | --- | --- |
| frontend, source, parser, AST, and Sema | `tests/compiler/frontend/` | `bench/compiler/frontend/` |
| CEIR common, text, semantic, planning, realization | `tests/compiler/ir/` | `bench/compiler/ir/` |
| profile artifacts and environments | `tests/compiler/profile/` | `bench/compiler/profile/` |
| discovery, composition, program, and planning compiler | `tests/compiler/planning/` | `bench/compiler/planning/` |
| lowering and realization services | `tests/compiler/realization/` | `bench/compiler/realization/` |
| backend ABI, CPU, NVCC, and NVPTX providers | `tests/compiler/backends/` | `bench/compiler/backends/` |
| object CEIR and cross-TU/LTO | `tests/compiler/lto/` | `bench/compiler/lto/` |
| libCellerator C/C++ API, stdlib, install, and package use | `tests/compiler/sdk/` | `bench/compiler/sdk/` |
| celleratord protocol, queries, and editor workflows | `tests/compiler/language_server/` | `bench/compiler/language_server/` |

Tests may use deeper subdirectories matching the owned public component. A test
that spans components lives with the highest layer whose behavior it asserts;
lower-layer fixtures remain independently tested by their owners. This avoids a
second miscellaneous integration tree with unclear responsibility.

The current `tests/compiler/a04/` gates validate the source-layout freeze itself.
They are bootstrap validation receipts and do not become a product subsystem.

## Test rules

- Exact reference and adversarial semantic tests live in the owning test path.
- ABI, serialization, stale-identity, order, capacity, and invalid-extension
  checks accompany the public contract they protect.
- GPU tests remain under the relevant backend family and state the required
  provider; host-only compiler tests must not acquire CUDA dependencies.
- Package-level consumption is SDK ownership, while executable command behavior
  is frontend/driver or language-server ownership according to the shared API
  invoked.
- Compatibility tests for migrated JBC mechanisms stay source-linked to the
  owning new compiler family and identify their historical reference.

## Benchmark rules

Benchmarks are not substitutes for correctness gates. Every benchmark records
the applicable toolchain, build mode, input shape, warmup/repeat counts,
included costs, and correctness result. GPU benchmarks additionally use the
repository benchmark mutex and active resource reservation.

The benchmark family determines the primary cost:

- `frontend/`: parse, source-map, bridge, and diagnostic latency/memory;
- `ir/`: construction, verification, transform, serialization, and artifact size;
- `profile/`: loading, validation, merge, and environment-query cost;
- `planning/`: discovery, composition, exact coverage, and whole-plan cost;
- `realization/`: lowering, resumption, and prepared-stage construction;
- `backends/`: code generation, compilation, execution, and complete conversion;
- `lto/`: object companion, cross-TU merge, and whole-program optimization;
- `sdk/`: consumer compile/link, resource lookup, and package footprint; and
- `language_server/`: cold start, incremental update, query latency, and memory.

Cross-family results may be summarized elsewhere, but raw harness and evidence
remain in the family that owns the measured mechanism.

## Existing source correspondence

Existing runtime, geometry, JBC, CE-EXOP, CE-GEO, and CE-LIVE tests and
benchmarks keep their current paths. This freeze adds compiler-specific owners;
it does not move historical evidence or make old benchmark reports normative
architecture.

## Compatibility and deferred work

This layout contract changes no test registration, benchmark target, runtime,
or JBC behavior. Central integration owns `tests/CMakeLists.txt` and
`bench/CMakeLists.txt`. No Part Two JIT or deep CellShard benchmark is added.

## Validation evidence

`tests/compiler/a04/freeze_compiler_test_and_benchmark_locations_test.cc`
verifies a one-to-one test/benchmark mapping for every planned family, the
source-layout bootstrap exception, evidence rules, and compatibility boundary.
