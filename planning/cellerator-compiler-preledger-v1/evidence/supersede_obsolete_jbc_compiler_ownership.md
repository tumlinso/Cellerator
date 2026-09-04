# Supersession record for obsolete JBC compiler ownership

Todo: `CE-CCP1-A01-007`

## Disposition

The original JBC preledger remains immutable historical evidence, but its
assignment of biological compiler layers to CellShard is superseded. Cellerator
now owns representative-profile evidence, discovery, independent exact
certification, atom semantics, typed composition and grammar, basis/no-basis
selection, global program IR, and portable schedule/ruleset compilation.

This is an ownership correction, not a deletion or clean-room rewrite. Useful
JBC implementations, tests, benchmarks, and evidence must move, split, or be
adapted with provenance under the no-code-loss migration policy.

## Trace from historical claims to current ownership

| Compiler layer | Historical JBC assignment | Current authoritative disposition |
| --- | --- | --- |
| Evidence and proposal discovery | `planning/jbc-preledger-v1/01_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md:11` assigned discovery of reusable execution atoms to CellShard; `:122` placed the overlapping evidence atlas there. `planning/jbc-preledger-v1/08_DEPENDENCY_INTEGRATION_AND_PARALLELISM_MAP.md:21-32` assigned the evidence core and ten discovery lanes to CellShard. | Cellerator owns representative-profile evidence and proposal discovery (`03_CELLERATOR_CELLSHARD_SUPERSESSION_AND_JBC_REHOMING.md:23-27`, migration row `:86`). |
| Independent exact certification | `planning/jbc-preledger-v1/06_EXACT_VALIDATION_AND_BIOLOGICAL_EVIDENCE_PLAN.md:5-9` declared the certification workstream a CellShard lane, and `:22-26` placed its code and tests below `CellShard/compiler/certification`. The dependency map repeats this at `08_DEPENDENCY_INTEGRATION_AND_PARALLELISM_MAP.md:22`. | Cellerator owns exact certification and exact coverage (`03_CELLERATOR_CELLSHARD_SUPERSESSION_AND_JBC_REHOMING.md:27-28`, migration row `:87`). Discovery remains nonauthoritative and certification remains independent. |
| Typed composition and grammar | The old invariant assigned composition grammar to CellShard (`01_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md:125`); the interface map assigned composition productions to CellShard (`02_INTERFACES_OWNERSHIP_AND_SOURCE_MAP.md:20`); the dependency map assigned composition and explicit/induced grammar lanes there (`08_DEPENDENCY_INTEGRATION_AND_PARALLELISM_MAP.md:33-35`). | Cellerator owns typed composition and grammar (`03_CELLERATOR_CELLSHARD_SUPERSESSION_AND_JBC_REHOMING.md:29`, migration row `:89`). Induced mechanisms remain experimental and may end in non-promotion. |
| Basis, no-basis, and superatoms | The old interface map assigned the biological execution basis manifest to CellShard (`02_INTERFACES_OWNERSHIP_AND_SOURCE_MAP.md:21`), while the dependency map assigned basis and superatom lanes there (`08_DEPENDENCY_INTEGRATION_AND_PARALLELISM_MAP.md:36-37`). The old architecture required a complete no-basis outcome at `01_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md:132`. | Cellerator owns basis/no-basis selection and any retained superatom promotion (`03_CELLERATOR_CELLSHARD_SUPERSESSION_AND_JBC_REHOMING.md:30-31`, migration rows `:90-91`). No-basis and measured non-promotion remain valid results. |
| Global operation/program IR | The old invariant assigned global decomposition and schedules to CellShard (`01_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md:125`); the dependency map assigned `CS-JBC-L-GLOBAL-IR` and the global graph compiler to CellShard (`08_DEPENDENCY_INTEGRATION_AND_PARALLELISM_MAP.md:40,145`). | Cellerator owns global operation/program IR and cross-operation planning (`03_CELLERATOR_CELLSHARD_SUPERSESSION_AND_JBC_REHOMING.md:33-34`, migration row `:94`). |
| Portable schedule compilation | The old interface map made the global operation provider and portable schedule a CellShard interface (`02_INTERFACES_OWNERSHIP_AND_SOURCE_MAP.md:24`), and the dependency map placed the portable schedule compiler in CellShard (`08_DEPENDENCY_INTEGRATION_AND_PARALLELISM_MAP.md:40`). | Cellerator owns portable schedule/ruleset compilation (`03_CELLERATOR_CELLSHARD_SUPERSESSION_AND_JBC_REHOMING.md:35`, migration row `:95`). |

## CellShard boundary retained

CellShard still owns concrete artifact storage and encoded replicas, dataset
sharding, materialization from Cellerator rules, payload staging and assembly,
placement, residency, transport, leases, recovery, runtime command execution,
and storage-oriented publication. It may contribute concrete capabilities and
external costs through generic interfaces, but it is not the semantic planner.
This retains the useful systems half of the old design without leaving compiler
meaning in the storage/runtime repository.

Part One reserves only the narrow compiled-ruleset/materialization-request seam.
Deep CellShard application and runtime integration remains deferred to Part Two.

## Historical preservation receipt

No file under `planning/jbc-preledger-v1/`, no closed Todo result, and no old run
identity is modified by this Todo. `CE-JBC-RUN-V1` is not reused. At the source
commit for this receipt, the principal historical documents had these hashes:

- `01_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md`: `59e10ad8963f703c1879957ee0fd5e92bc0cc0a9bf164c3ef81b114bb0c9231d`
- `02_INTERFACES_OWNERSHIP_AND_SOURCE_MAP.md`: `53f94c42f92fddda6ccf1e1f6bd0134897c5f96887fa4241a44d420ed8f5e023`
- `04_CELLSHARD_ATOM_EVIDENCE_COMPILER_PLAN.md`: `22232440f9fb5c172d51b4028418be4190b297b7615b64f493fe9178a1f210bf`
- `06_EXACT_VALIDATION_AND_BIOLOGICAL_EVIDENCE_PLAN.md`: `41092f5e1cbd170c3abd735568c23b7f10b878eac4e9f30810587265dcb59b1a`
- `08_DEPENDENCY_INTEGRATION_AND_PARALLELISM_MAP.md`: `702e98f5308f8961024a5e28707782af584374253751823975ecaa61fbd12134`

Future migration is governed by
`planning/cellerator-compiler-preledger-v1/inventories/jbc_source_migration.csv`.
Each moved source retains its repository, branch, commit, original path,
recoverable Todo, tests/evidence, SHA-256, disposition, and target-path receipt.
