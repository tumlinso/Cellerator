# Frozen architecture-reconciliation receipt

Todo: `CE-CCP1-A01-009`
Interface: `CE-CCP1-I01-AUTHORITY-BASELINE` version 1

## Frozen boundary

This receipt aggregates the authoritative source cursor, language and IR
document hashes, accepted Todo schema, exact CLI discovery, Part One charter,
JBC ownership supersession, and identifier/artifact collision audit that all
later workstreams must consume.

At this frozen reconciliation boundary, Part One product implementation and
post-apply plan-semantic Todo mutation have not begun. The plan had already
been applied and activated, and A01 validation Todos necessarily advanced live
workflow lifecycle, claim, gate, evidence, and projection state. Those expected
transactional changes are not compiler implementation and do not alter the
declared plan semantics. No source, compiler, runtime, historical result, task
definition, dependency, scope, interface contract, checkpoint, barrier, lane,
lock, or integration topology was changed while producing this baseline.

## Authority cursor

- Project: `cellerator`
- Project UUID: `0ccaac37-dbbf-448e-a5f8-def197a70aba`
- Run: `CE-CCP1-RUN-V1`
- Reconciled baseline Todo/workflow revision: `3904`
- Reconciled semantic fingerprint:
  `fb53d368f34add25316a3aab81251f9dd19d43dcc5c4a672f1054cd271de3837`
- Repository baseline observed by Project Control:
  `31e491ed29de0fcde70259cbeab8c5c7ad353485`
- Provider revision skew: zero across export, semantic state, status, and
  semantic workflow.

The authority cursor was obtained from fresh Project Control reads. It is a
historical baseline, not an assertion that live lifecycle revision remains
3904 after A01 execution.

## Source and specification identities

| Authority | SHA-256 |
| --- | --- |
| `docs/language/cellerator-language-specification.md` | `22329eef2e84e6b48d7d304e77c2ed65b75f29721d65f499680faa08c8efe15b` |
| `docs/language/cellerator-programming-guide.md` | `5fd78c016fddf3876e20c291affd9042f9dfc854ac520c0834dcb20663e20c26` |

Those were the only regular files under `docs/language` at capture time. No
separately supplied external IR document was available; the in-repository
language documents and the Part One Semantic/Planning/Realization IR plans are
the reviewable authority set.

The embedded CellShard gitlink and independent sibling checkout both resolved
to `b9749ad3e5146a04f847533d8c6f1a54146aed20` at the non-atomic Git observation.
The source checkout also retained all registered historical JBC worktrees; the
missing `/tmp/ce-jbc-main-delivery-20260901` entry was recorded as prunable,
not silently removed.

## Plan and workflow contract

- Applied program source: schema 3, 557 tasks, 44 checkpoints, 512 gates,
  10 barriers, 41 interfaces, 38 lanes, 11 locks, one active run.
- Accepted plan schemas: compatibility schema 2 and first-class workflow
  schema 3.
- Database migration: 10; workflow snapshot section: 1; workflow protocol: 2.
- Exactly one root lane is required; child lanes name `parent_lane_id`.
- Project Control is the workflow front door. `next_task` is the authoritative
  claim operation; there is no separate `claim_task` tool.
- Project Control plan validation is preview/read-only. Applying schema 3
  creates the active run; there is no separate `plan activate-run` command.
- Direct Todo commands documented by the CLI audit are lower-level maintenance
  contracts, not permission to bypass Project Control.

The collision audit found no duplicate task, interface, checkpoint, barrier,
lane, run, lock, gate, or decision identifiers and no cross-namespace exact
collisions. Seven repeated produced paths were reviewed as deliberate serial or
integration ownership, not competing concurrent artifacts.

## Frozen architectural reconciliation

1. Cellerator owns the compiler, source semantics, all public CEIR, profiles,
   discovery, exact certification, atom semantics, composition/grammar,
   basis/no-basis, decomposition, global program IR, planning, portable
   schedules/rulesets, realization, lowering, compiler tooling, libCellerator,
   SDK/standard-library foundations, and celleratord.
2. Useful JBC code, tests, measurements, and evidence are preserved and moved,
   split, adapted, or wrapped with provenance. `CE-JBC-RUN-V1` and its completed
   records remain historical and are not reused or rewritten.
3. CellShard retains concrete artifacts, storage, sharding, materialization,
   staging, placement, residency, transport, leases, recovery, and runtime
   commands. It may provide generic external costs/capabilities but is not the
   semantic planner.
4. Part One defines only narrow compiled-ruleset/materialization, external-cost,
   lowering-resumption, and compatibility seams. General JIT and deep CellShard
   runtime/materialization evolution remain Part Two.
5. Public CEIR comprises writable Semantic, Planning, and Realization IR with
   canonical text, sectioned binary forms, exact round trips, reflection,
   extensions, replacement passes, and removable cold provenance.
6. Cellerator retains ordinary C++ pointer, reference, template, custom-layout,
   CUDA, PTX, native-code, unsafe, forced, raw, and manual control wherever the
   relevant contract technically permits it. Pointer-free layout is required
   only by specific persistent or relocatable artifacts.
7. Activated biological compilation requires an explicitly bound
   representative profile; ordinary C/C++ fallthrough and structural tooling
   do not silently receive a generic profile.
8. Performance is governed by complete measured cost, exact correctness, and
   explicit fallbacks. Evaluated-not-promoted is a legitimate experimental
   result.

## Evidence manifest

The following SHA-256 values bind the eight human-reviewable A01 receipts that
form this aggregate. They were recomputed immediately before this receipt was
authored.

| Evidence file | SHA-256 |
| --- | --- |
| `capture_the_reconciled_project_control_authority_cursor.md` | `41b70efc8d4608443871af6e3ec2fae099426569a3504858abac12fa54ef851a` |
| `capture_cellerator_and_embedded_cellshard_git_identities.md` | `dfeb553bd8de617f0fd092985080baa5ea876626f8fd21f9a914734f61ffde29` |
| `hash_and_index_every_language_and_ir_authority_document.md` | `306bfa53f9cce0508eaf1c1a2f9db566f2e616887c1971c8eb13f73c0604ed96` |
| `audit_accepted_todo_plan_schemas_and_live_precedents.md` | `1998531ec218c650894a990a0112ae8669b3875fa95f4cd4e8259aa3c6636f07` |
| `discover_and_preserve_exact_todo_cli_help.md` | `ce181a3c1de4b014c0b46a116227668474dd96bc8b8a1d63ff3e229553c78838` |
| `publish_the_part_one_compiler_charter.md` | `2f90b49d51bc4b1660c6a2d9b7afd4190217a3995310c4c355eca30a5f654e41` |
| `supersede_obsolete_jbc_compiler_ownership.md` | `a68b1e0a695526b37c7891dcdd998c02cdd5018fa58ce3e42f4b327b74e3ddae` |
| `audit_program_identifier_and_artifact_name_collisions.md` | `93ebe7e4417d0862c49f6a24767c514182e438f32ce777f8c96173f3b77081d8` |

Any later workstream that observes a different source hash or conflicts with
this ownership split must stop at its declared interface/checkpoint and publish
an explicit versioned reconciliation. It must not silently reinterpret this
baseline or mutate its historical inputs.
