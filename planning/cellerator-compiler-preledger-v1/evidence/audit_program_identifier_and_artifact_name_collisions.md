# Program identifier and artifact-name collision audit

Todo: `CE-CCP1-A01-008`

## Result

The deterministic audit found **zero unresolved identifier or artifact-name
collisions**. The applied Part One program uses disjoint, unique identifiers for
tasks, interfaces, checkpoints, barriers, lanes, runs, locks, gates, and
decisions. Existing wire magics remain reserved. New profile and CEIR artifact
identities are owned by later freeze tasks and must select values outside that
reserved set.

The authority observation for this audit is Project Control revision `3934`,
where `CE-CCP1-A01-008` was the active A01 claim. The source manifest checked was
`planning/cellerator-compiler-preledger-v1/machine/cellerator-compiler-part1.todo-plan.json`
(schema 3), which is the applied program's source representation; Project
Control remains the live workflow authority.

## Deterministic identifier audit

Each namespace was collected in source order, counted, converted to a set, and
checked for repeated values. Every namespace also participated in an exact
cross-namespace intersection check.

| Namespace | Declared | Unique | Internal duplicates |
| --- | ---: | ---: | ---: |
| Tasks | 557 | 557 | 0 |
| Interfaces | 41 | 41 | 0 |
| Checkpoints | 44 | 44 | 0 |
| Barriers | 10 | 10 | 0 |
| Lanes | 38 | 38 | 0 |
| Runs | 1 | 1 | 0 |
| Locks | 11 | 11 | 0 |
| Gates | 512 | 512 | 0 |
| Decisions | 2 | 2 | 0 |

Exact identifier intersections across those namespaces: `0`.

The prefixes also communicate kind rather than relying only on database type:
`CE-CCP1-` tasks, `CE-CCP1-I` interfaces, `CE-CCP1-CP-` checkpoints,
`CE-CCP1-B-` barriers, `CE-CCP1-L-` lanes, `CE-CCP1-RUN-` runs, and
lowercase `ce-ccp1-` lock names. The historical `CE-JBC-*`, `CS-JBC-*`,
`CE-PTR-*`, and `CE-EXOP-*` families remain distinct and are not reused.

## Produced-path overlap review

Produced artifact paths are not identifiers: integration tasks deliberately
aggregate central files, and serial tasks may incrementally develop a common
component. The plan has 1,619 produced-path declarations and 1,580 unique
paths. All seven repeated paths are resolved by declared topology:

| Path | Resolution |
| --- | --- |
| `CMakeLists.txt` | Thirteen J03 final-audit tasks are serial in one lane and operate under the central integration lock. |
| `include/Cellerator/Cellerator.hh` | The same thirteen J03 tasks serially audit/finalize the umbrella; this is intentional integration ownership. |
| `src/compiler/CMakeLists.txt` | Twelve B01 build-partition tasks are serial in one lane and feed their integration checkpoint. |
| `include/Cellerator/compiler/api/define_c_compiler_session_api_v1.hh` | H02-002 establishes the C API and H02-003 adds the adjacent C++ API serially in the same lane. |
| `include/Cellerator/sdk/define_c_compiler_session_api_v1.hh` | Same H02-002 to H02-003 serial handoff. |
| `src/compiler/api/define_c_compiler_session_api.cc` | Same H02-002 to H02-003 serial handoff. |
| `tests/compiler/h02/define_c_compiler_session_api_test.cc` | Same H02-002 to H02-003 serial test evolution. |

These overlaps are not concurrent exclusive ownership and do not create two
artifact names or wire identities. Integration lanes remain the sole owners of
shared central aggregation.

## Existing artifact namespace reservations

The source audit found these current Cellerator binary identities:

| Logical name | Stored magic | Source |
| --- | --- | --- |
| Packing-plan image | `CPI1` (`0x31495043`) | `include/Cellerator/geometry/packing_plan.hh` |
| Sample selection | `CPS1` (`0x31535043`) | `include/Cellerator/compute/sampling.hh` |
| Sampled CSR image | `SCR1` (`0x31524353`) | `include/Cellerator/compute/sampling_materialization.hh` |
| Persistent packing payload / CPK1 | `CELLPK01` | `src/geometry/persistent_packing_payload.cc` |
| Semantic geometry / CSG1 | `CELLCSG1` | `src/geometry/persistence/semantic_geometry_image_v1.cc` |
| Execution image / CPE2 | `CELLEX02` | `src/geometry/persistence/execution_image_v2.cc` |
| External oracle snapshot | `CEORCL1\0` | `src/geometry/optimizer/oracle/external_snapshot.cc` |
| Chunk manifest v1 | `CCECHNK1` in little-endian byte order | `include/Cellerator/geometry/persistence/chunk/chunk_manifest_v1.hh` |

Historical planning also reserves the established names `CSH5`, `CSPACK`, and
`CPEXEC01`; they are compatibility/storage formats and must not be repurposed
for profiles or CEIR.

## Profile and CEIR artifact names

- `.ceprofile` is the source-language documentation's current human-facing
  profile filename suffix. It is distinct from every current source and binary
  artifact name. `CE-CCP1-D03-001` owns the final profile container name,
  version, and magic and must audit the reserved table above before freezing it.
- `CE-CCP1-I15-CEIR-TEXT` uniquely names the common CEIR textual/binary
  artifact interface. `CE-CCP1-D01-010` and `CE-CCP1-D01-014` own the binary
  container and compatibility freeze. No CEIR binary magic or mandatory
  filename suffix is prematurely assigned here.
- Semantic, Planning, and Realization IR are abstraction-level markers inside
  the common CEIR family, not three aliases for CSG1, CPE2, CPK1, or the profile
  container.
- Cellerator source activation remains `#pragma cellerator`; a filename suffix
  is never the semantic switch.

## Collision policy

Later format owners must fail closed if a proposed name, magic, interface ID, or
extension collides with this reserved namespace or with live authority. They
must add a versioned adjacent identity rather than changing an existing wire
meaning. Historical identifiers stay reconstructable, central-file sharing
must follow declared integration topology, and no Part Two format is reserved
or implemented by this audit.
