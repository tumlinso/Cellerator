# JBC Tests and Benchmarks Mapped to Preserved Behavior

This receipt completes `CE-CCP1-A02-005`. It indexes every current JBC test,
test helper, benchmark source, benchmark record, and benchmark disposition in
Cellerator and embedded CellShard. The mapping preserves behavior evidence; it
does not imply that an experimental mechanism was promoted.

## Frozen evidence set

The source set contains 430 repository files:

- 94 Cellerator files under `tests/jbc/`;
- 2 Cellerator files under `bench/jbc/`;
- 328 CellShard files under `tests/jbc/`;
- 6 CellShard files under `bench/jbc/`.

Prefixing paths with `CE:` or `CS:`, sorting, and newline-terminating the list
produces SHA-256
`a8bdbe583ff48762b68fb8ae550d0f834148cfeb14ff462271ad09c2cd14aa6b`.
The focused gate rebuilds this set directly from both repositories.

## Behavior index

Every evidence-set file has exactly one primary protected behavior, selected by
its owning test/benchmark directory. Counts include helpers and data records as
well as executable sources.

| Repository behavior | Files | Evidence paths | Reusable subsystem protected |
|---|---:|---|---|
| CE semantic interfaces | 12 | `tests/jbc/interfaces/**` | CE semantic interfaces |
| CE atom-fragment preparation | 14 | `tests/jbc/fragment/**` | CE atom-fragment preparation |
| CE decomposition catalog | 18 | `tests/jbc/decomposition/**` | CE decomposition catalog |
| CE atom/value planes | 10 | `tests/jbc/atom_plane/**` | CE atom/value planes |
| CE multi-extent binding | 10 | `tests/jbc/multi_extent/**`, `bench/jbc/multi_extent/**` | CE multi-extent binding and candidate |
| CE external complete-cost exchange | 6 | `tests/jbc/external_cost/**` | CE external complete-cost exchange |
| CE lowering resumption | 10 | `tests/jbc/resumption/**` | CE lowering resumption |
| CE cross-operation reuse | 9 | `tests/jbc/cross_operation/**`, `bench/jbc/cross_operation/**` | CE cross-operation validation scenarios |
| CE verification and package gates | 7 | `tests/jbc/verification/**` | CE aggregate/package surface; CE independent verifier helpers |
| CS atom model | 20 | `tests/jbc/atom/**` | CS atom model |
| CS atom store | 30 | `tests/jbc/atom_store/**` | CS atom store |
| CS basis selection | 17 | `tests/jbc/basis/**` | CS basis selection |
| CS exact certification | 16 | `tests/jbc/certification/**` | CS exact certification |
| CS composition | 24 | `tests/jbc/composition/**` | CS composition |
| CS evidence atlas | 16 | `tests/jbc/evidence/**` | CS evidence atlas |
| CS global graph and schedule | 14 | `tests/jbc/global_ir/**` | CS global graph and schedule |
| CS explicit grammar | 10 | `tests/jbc/grammar/explicit/**` | CS explicit grammar |
| CS induced grammar | 12 | `tests/jbc/grammar/induced/**`, `bench/jbc/grammar/**` | CS induced grammar experiment |
| CS persistent partials | 18 | `tests/jbc/partial/**` | CS persistent partials |
| CS runtime v2 | 26 | `tests/jbc/runtime/**`, `bench/jbc/runtime/**` | CS runtime v2 |
| CS superatoms | 8 | `tests/jbc/superatom/**` | CS superatoms |
| CS integrated validation | 36 | `tests/jbc/validation/**` | CS integrated validation/package matrix |
| CS bicluster discovery | 9 | `tests/jbc/discovery/bicluster/**`, `bench/jbc/bicluster/**` | CS bicluster discovery |
| CS co-support discovery | 11 | `tests/jbc/discovery/co_support/**` | CS co-support discovery |
| CS factor/topic discovery | 6 | `tests/jbc/discovery/factor_topic/**` | CS factor/topic discovery |
| CS motif discovery | 8 | `tests/jbc/discovery/motif/**` | CS motif discovery |
| CS multimodal discovery | 10 | `tests/jbc/discovery/multimodal/**` | CS multimodal discovery |
| CS operation-trace discovery | 8 | `tests/jbc/discovery/operation_trace/**` | CS operation-trace discovery |
| CS overlap discovery | 6 | `tests/jbc/discovery/overlap/**` | CS overlap discovery |
| CS sequence compatibility | 6 | `tests/jbc/discovery/sequence_compat/**` | CS sequence compatibility discovery |
| CS support-signature discovery | 10 | `tests/jbc/discovery/support_signature/**` | CS support-signature discovery |
| CS trajectory discovery | 13 | `tests/jbc/discovery/trajectory/**`, `bench/jbc/trajectory/**` | CS trajectory discovery |

The 32 behavior counts sum to 430. Every subsystem marked reusable (`complete
and tested`) by A02-004 has at least one mapped test or evidence group.

## Primary evidence-form index

Evidence files often exercise several modes. The primary form below is a
non-overlapping inventory classification, not a claim that a unit test cannot
also contain negative cases.

| Primary form | Files | Deterministic selection | What it protects |
|---|---:|---|---|
| unit test | 352 | Remaining `tests/jbc/**` files | Local contract behavior, identity, ordering, algebra, lifecycle, and candidate mechanics. |
| property test | 48 | Validation/verification trees and names containing `exact_`, `stability`, or `property`, after higher-priority rules | Cross-fixture invariants, exact reconstruction, stable evidence, portable consumption, and end-to-end slices. |
| malformed-input test | 8 | Names containing `fault`, `recovery`, `invalid`, `malformed`, `mismatch`, `overflow`, `duplicate`, or `stale` | Fail-closed validation, torn/corrupt state, duplicate coverage, invalidation, and recovery outcomes. |
| promotion evidence | 14 | Test basenames containing `promotion`; validation ablation/null-transform files; exact-oracle null benchmark test | Evidence gates, matched nulls, ablations, and explicit promotion/demotion decisions. |
| benchmark fixture | 7 | All `bench/jbc/**` except the CE X08 disposition | Complete-cost comparison inputs, assembly comparison, induced grammar, trajectory null, bicluster, and NUMA/GPU process-model evidence. |
| non-promotion result | 1 | `bench/jbc/cross_operation/CE-JBC-X08.md` | Explicitly records a conservative decision procedure with no hardware-performance claim. |

Additional non-promotion evidence is source-linked at
`components/CellShard/docs/JBC/evidence/biological_novelty_readiness.md`: the
integrated implementation is technically ready, but no new reserved biological
campaign was run and no biological-performance claim is promoted.

## Representative adversarial and performance anchors

- `tests/jbc/verification/numerical_verifier_v1_test.cc` reconstructs 4,096
  fragments exactly, then proves tolerance and duplicate-write failures.
- `components/CellShard/tests/jbc/validation/atom_store_faults_test.cc` checks
  the complete atom-store fault matrix and pinned-atom recovery outcome.
- `components/CellShard/tests/jbc/validation/null_transform_test.cc` and the
  paired atom/compiler ablations protect biological-mechanism attribution.
- `components/CellShard/bench/jbc/runtime/results/2026-09-01-dual-numa-process-campaign.json`
  records the controller reservation, benchmark mutex, four V100 identities,
  compiler/CUDA versions, fork-plus-exec correction, and correctness-only
  disposition; it makes no throughput claim.
- `components/CellShard/docs/JBC/evidence/integration_receipt.md` binds the full
  304-test host and sanitizer matrix, host benchmarks, CUDA builds, package
  consumer, and integration gates to the preserved CellShard history.
