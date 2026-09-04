# Duplicated JBC compiler mechanisms across repositories

This receipt compares the frozen JBC interface map with the implemented
Cellerator and CellShard source in the A02 read scope. It identifies semantic
overlap rather than relying on filenames alone. Each overlap has an explicit
merge, adapter, or retain-distinct decision so the Part One rehome cannot create
blind parallel copies.

## Decision rules

- **MERGE-CELLERATOR**: converge compiler semantics and validation into one
  Cellerator-owned contract. Preserve useful CellShard implementation while it
  is migrated; do not create a third representation.
- **ADAPT-CELLSHARD**: keep only a narrow fieldwise/storage/runtime adapter in
  CellShard. The adapter may validate representation and transport preconditions
  but must not redefine compiler meaning.
- **RETAIN-DISTINCT**: the records touch similar concepts but have different
  owners and lifetimes. Keep both, with an explicit conversion or identity link.

## Duplicate and overlap decisions

| Duplicate | Mechanism | Cellerator source | CellShard source | Semantic overlap and difference | Decision |
|---|---|---|---|---|---|
| JBC-D01 | Namespaced persistent identity | `execution/joint_compiler/persistent_identity_v1.hh` | `compiler/atom/persistent_identity_v1.hh` | Exact 16-byte `{producer_namespace, local_identity}` value and 24-byte record are duplicated, including nonzero validation and ordering. | **MERGE-CELLERATOR + ADAPT-CELLSHARD:** Cellerator type/validator becomes canonical; CellShard uses a fieldwise versioned adapter until consumers migrate. |
| JBC-D02 | Profiling stable identity | `profiling/joint_compiler/manifest_v1.hh` | `compiler/atom/persistent_identity_v1.hh` and evidence identities | Cellerator profiling has another two-`uint64_t` `stable_identity_v1`, but it accepts either half nonzero, unlike the joint-compiler identity. | **MERGE-CELLERATOR:** new compiler/profiling records use `persistent_identity_v1`; preserve a checked legacy manifest adapter and its weaker historical validity rule only for old inputs. |
| JBC-D03 | Exact logical coverage | `execution/joint_compiler/logical_coverage_v1.hh` | `compiler/atom/logical_coverage_v1.hh`, `compiler/composition/coverage_v1.hh`, certification coverage headers | CellShard mirrors Cellerator schema/kind/role/record bytes and points back to caller-owned Cellerator coverage, while composition/certification add operations over coverage. | **MERGE-CELLERATOR + ADAPT-CELLSHARD:** one Cellerator coverage algebra and validator; CellShard retains a pointer-first storage/application reference, never an independent semantic coverage definition. |
| JBC-D04 | Atom semantic envelope | `atom_requirement_v1.hh`, `atom_affordance_v1.hh`, `logical_coverage_v1.hh`, decomposition contracts | `compiler/atom/common_atom_v1.hh` and plane directory headers | Common atom repeats identity, exact coverage, ports, planes, parents, evidence, affordance, and lineage around Cellerator-defined semantics. | **MERGE-CELLERATOR + ADAPT-CELLSHARD:** move semantic envelope/plane vocabulary to Cellerator; retain a CellShard materialization view that references canonical descriptors and owns no compiler policy. |
| JBC-D05 | Evidence identity | `persistent_identity_v1.hh`; `execution_export_v2.hh` evidence/build/device identities | `compiler/evidence/atom_evidence_record_v1.hh` | `evidence_identity_v1` is a third exact two-`uint64_t` namespaced identity with its own equality/order validator. | **MERGE-CELLERATOR:** replace new evidence identities with the canonical persistent identity; use a fieldwise reader for the frozen 80-byte CellShard record. |
| JBC-D06 | Evidence atlas and execution evidence | `profiling/joint_compiler/manifest_v1.hh`, `execution_export_v2.hh` | `compiler/evidence/evidence_atlas_v1.hh` and associated evidence tables | Both bind atom identities, generations, provenance/freshness, and evidence records. The atlas holds proposal observations; the export holds execution correctness/performance evidence. | **MERGE-CELLERATOR + RETAIN-DISTINCT:** share canonical evidence identity, provenance, freshness, and query validation in Cellerator, while retaining proposal-atlas and execution-export record kinds because proposals never certify exact coverage. |
| JBC-D07 | Complete cost and resource records | `planner/external_cost/{vector_v1,complete_cost_v1,frontier_v1,compiler_exchange_v1}.hh` and fragment complete cost | `compiler/composition/superatom/cost.hpp`, basis utility/budget records, `compiler/graph/physical_realization.hh` | Multiple records price preparation, persistent/transient bytes, movement, launches, latency, throughput, and reuse, but only the Cellerator exchange composes external and local complete cost. | **MERGE-CELLERATOR + ADAPT-CELLSHARD:** Cellerator complete-cost schema and frontier are canonical; CellShard reports measured storage/materialization/transport terms through the external-cost adapter. |
| JBC-D08 | Logical graph nodes and operation providers | Fragment request/result, requirement/affordance, and `compute/decomposition/provider_registry_v1.hh` | `compiler/graph/{operation_node,operation_provider,atom_dependency,access_effect}.hh` | Both describe typed operation inputs/outputs, provider capability, dependency, legal decomposition, and candidate selection. CellShard graph records currently use local strong IDs. | **MERGE-CELLERATOR:** rehome logical graph/provider semantics into Cellerator CEIR planning and translate legacy strong IDs fieldwise; CellShard receives a finalized schedule rather than owning provider choice. |
| JBC-D09 | Recipe expansion and lowering resumption | `execution/joint_compiler/lowering_resumption_v1.hh` and fragment frontier | `compiler/graph/graph_recipe.hh`, composition/grammar DAGs | All represent bounded resumable transformation from semantic operations toward executable stages. Recipe opcodes and DAGs are compiler policy; resumption additionally records bypassed phases and external availability. | **MERGE-CELLERATOR:** form one versioned recipe/DAG/resumption model in the compiler and preserve legacy readers; do not copy graph recipe logic beside the CEIR lowering pipeline. |
| JBC-D10 | Derivation DAG validators | no second Cellerator implementation yet; destination is compiler grammar/composition | `compiler/composition/derivation_dag_v1.hh` and `compiler/grammar/derivation_dag_v1.hh` | CellShard itself has two topological-sort/cycle validators with different node models, index widths, capacity rules, and ordering requirements. | **MERGE-CELLERATOR:** share one bounded DAG validation engine with typed composition and grammar record adapters; preserve 256-node/1024-edge composition limits where that ABI requires them. |
| JBC-D11 | Partial-result algebra and persistent partial | `compute/decomposition/partial_result_algebra_v1.hh`, concrete partial state/decomposition headers | `compiler/partial/partial_atom_v1.hh`, partial state headers, `compiler/certification/partial_result_compatibility_v1.hh` | Algebra identities, numerical/order policy, contribution coverage, state shape, and compatibility are repeated; CellShard adds persistence class and materialization generations. | **MERGE-CELLERATOR + ADAPT-CELLSHARD:** Cellerator owns algebra/state/coverage compatibility; CellShard stores a versioned partial payload envelope and validates storage generations against canonical metadata. |
| JBC-D12 | Physical realization and affordance | `atom_affordance_v1.hh`, fragment result, `execution_export_v2.hh` stages | `compiler/graph/physical_realization.hh` | Both bind logical candidates to provider/projection identities and account for preparation, persistent/transient bytes, and launches. | **MERGE-CELLERATOR + ADAPT-CELLSHARD:** planner selection and stage graph live in Cellerator; CellShard materializes the selected image and returns concrete residency/application handles. |
| JBC-D13 | Portable schedule versus execution export | `execution_export_v2.hh` | `compiler/schedule/portable_artifact.hh` | Both reference a graph/stages and identities, but export is compiler evidence while the portable artifact is a concrete launch/copy/barrier/transform/publish command stream. | **RETAIN-DISTINCT + ADAPT-CELLSHARD:** Cellerator emits the selected semantic schedule/export; CellShard owns the concrete portable command artifact and links it by digest without re-planning. |
| JBC-D14 | Distributed exact-coverage certificate | Export correctness receipt, exact coverages, decomposition contributions | `compiler/schedule/distributed_certificate.hh` and certification headers | Both attest coverage completeness and bind graph/contribution identities. CellShard certificate also carries participant/route realization. | **MERGE-CELLERATOR + ADAPT-CELLSHARD:** Cellerator owns the exact distributed semantic certificate; CellShard appends route/participant realization evidence without changing certified coverage. |
| JBC-D15 | Mirrored semantic validators | `validate_*` functions across joint compiler, decomposition, cost, and export | parallel `validate_*` functions across atom, evidence, composition, partial, graph, and schedule | Mirrored records repeat schema, identity, generation, pointer/count, ordering, capacity, coverage, and freshness checks, which can drift independently. | **MERGE-CELLERATOR + ADAPT-CELLSHARD:** one canonical semantic validator per contract; boundary validators check wire/storage shape then invoke or compare against the canonical rule. Independent semantic acceptance rules are forbidden. |
| JBC-D16 | Generation and freshness vocabulary | typed value/structure identities and export performance freshness | plain `uint64_t` atlas, lineage, partial, store, schedule, and residency generations | Both sides distinguish epochs, but duplicated untyped fields make accidental structure/value/materialization substitution possible. | **MERGE-CELLERATOR + ADAPT-CELLSHARD:** canonical typed compiler generations stay in Cellerator; CellShard retains store/materialization/residency incarnations and converts only explicitly paired fields. |

## Required convergence order

1. Freeze the Cellerator persistent identity, logical coverage, generation, and
   validation vocabulary before moving dependent atlas, graph, cost, partial,
   grammar, basis, or schedule code.
2. Rehome useful CellShard compiler implementations behind those contracts;
   compare their existing tests before removing any legacy include.
3. Introduce narrow CellShard adapters for durable store records, portable
   commands, materialization, residency, topology, and transport.
4. Remove a duplicate definition only after every consumer is on the canonical
   contract and the compatibility reader proves old artifacts still validate.

## Blind-copy prohibition

Part One must not create parallel Cellerator copies of these CellShard headers
and leave both writable. A migration patch must name the duplicate ID above,
select its canonical owner, preserve or adapt its tests, and either delete the
old semantic definition after convergence or mark it as a versioned forwarding
compatibility surface. CellShard-only store/runtime state remains distinct;
similar field names alone are not authority to absorb it into the compiler.
