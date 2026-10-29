# CE-LIVE candidate and projection activation inventory

CE-LIVE-13 established the audit of the current operation-core candidates and
projection preparation seams. CE-LIVE-21 now activates that inventory as the
deterministic, allocation-free host catalog in
`include/Cellerator/compute/math/operation_core/builtin_catalog.hh`. It is not
a planner, stable persistence ABI, runtime owner, or replacement operation
framework. The original machine-readable audit remains
`bench/ce_live/catalog/candidate_inventory_v1.json`.

All forward candidates consume the CE-LIVE-11 relation orientation: feature or
gene source to row, cell, or module destination. Transpose/backward retains the
same structure identity, epoch, and logical edges and uses CTP1 plus an explicit
transpose value-position map.

## Actual operation-core candidates

| Candidate | Operation and regime | Projection | Numeric tuple | Capabilities | Persistent preparation | Output |
|---|---|---|---|---|---|---|
| `row_masked_n1_candidate` | weighted relation reduce, N=1 | device-rebound CPK1 schema 1, `native_row_masked` | configured `real::storage_t` / `compute_t` / `accum_t`; scalar u32 | deterministic, graph capture, persistent preprocessing | caller-owned `row_masked_n1_prepared_state` via `prepare_row_masked_n1_operation` | preserve supplied row axis, overwrite, zero workspace |
| `csr_fallback_candidate` | weighted relation reduce, N=1 | preconstructed `execution_csr_view` schema 1 | f16 sparse; f32 dense/output/multiply/accumulate; scalar u32 | deterministic, persistent preprocessing; no graph claim | caller-owned `csr_fallback_prepared_state` via `prepare_csr_fallback_operation` | preserve supplied row axis, overwrite, zero workspace |
| `feature_major_small_n_candidate` | SpMM, 1 <= N <= 16 | FMP1 schema 1 variant 1, `native_feature_major` | f16 sparse; f32 dense/output/multiply/accumulate/scalar | deterministic, graph capture, persistent preprocessing | caller-owned `feature_major_small_n_prepared_state` via `prepare_feature_major_small_n_operation` | preserve row and dense-column axes, overwrite, zero workspace |
| `feature_major_cta_medium_n_candidate` | SpMM, 17 <= N <= 64 | the same FMP1 bytes; distinct CTA schedule ID | same f16/f32 tuple | deterministic, graph capture, persistent preprocessing | same state type via `prepare_feature_major_cta_medium_n_operation` | preserve row and dense-column axes, overwrite, zero workspace |
| `transpose_backward_n1_candidate` | transposed SpMM/backward, N=1 | CTP1 schema 1 variant 1 over a distinct FMP1 forward projection | f16 sparse; f32 dense/output/multiply/accumulate/scalar | deterministic, graph capture, persistent preprocessing | caller-owned `transpose_backward_prepared_state` via `prepare_transpose_backward_n1_operation` | preserve feature and dense-column axes, overwrite, zero workspace |

Every row above has a candidate factory and an individual registration helper.
No source currently registers all five as built-ins. Registration alone is not
activation: each candidate's generic `prepare` callback assumes its typed helper
has already validated and installed candidate-specific persistent state.

## Physical projection inventory

- CPK1 is the compatible native row-masked payload. Its durable geometry,
  values, and canonical recovery maps remain owned by CellPack; direct execution
  aliases validated, device-rebound bytes.
- FMP1 is pointer-free feature-major schema 1 variant 1. The warp and CTA
  candidates share this physical projection; schedule identity does not invent
  another semantic relation or projection format.
- CTP1 is pointer-free transpose schema 1 variant 1. It names both its own and
  its forward FMP1 projection identity and carries forward-value and exact
  logical-edge maps.
- `execution_csr_view` schema 1 is a runtime typed view, not a new durable CPE2
  payload schema. Its construction and transfer are visible preparation cost.
- CPE2 schema 2 can describe native row-masked, native feature-major, CTA
  macrotile, dense fragment, CSR, SELL, BSR, Blocked-ELL, vendor-specific,
  transpose/backward, and architecture-specific entries. Enumeration is not
  implementation. CTA currently executes FMP1, while dense fragment, SELL,
  BSR, Blocked-ELL, vendor-specific, and architecture-specific have no retained
  built-in operation-core candidate.

## Evidence boundaries

The feature-major CTA source records V100 evidence for a 65,536 by 32,768
structure with 2,097,152 f16 values and f32 operands at N=17, 32, and 64. CTA
won every tested N=32/64 regime and high-sharing N=17; CSR won low-sharing N=17
and row-masked won medium-sharing N=17. This is regime evidence, not a global
default, and the declared eight-use horizon can retain row-masked after
construction cost.

The transpose candidate has independent correctness across two mutable value
generations but makes no performance-promotion claim. The native training slice
is an implemented N=16 FMP1/CTP1 composed workflow outside `candidate_registry`;
its source records a complete-step V100 win over persistent generic CSR for its
tiny measured fixture. It cannot be silently treated as a built-in candidate.
The custom CSR fallback is not cuSPARSE. Retained CP-Math v1 CSR/BELL code is
explicit compatibility evidence and must not be registered as a second runtime.

`sequence_predicate_accumulate` exists as an operation kind, but no matching
operation candidate exists in the audited math source. Sequence integration
therefore remains an activation gap rather than an inferred built-in.

## Minimal future host-side activation contract

CE-LIVE-21 may implement one allocation-free host catalog adjacent to the
operation core. It must compose the existing pieces instead of widening
`operation_candidate` or inventing a second registry. Each catalog descriptor
needs only:

1. the existing candidate factory or registration function;
2. a catalog-local preparation-family tag and typed bridge function;
3. the accepted projection kind, schema, variant, and required prebound typed
   view;
4. the exact numeric tuple and bounded dense-width regime;
5. persistent-state size/alignment and caller-owned storage requirements;
6. output-axis/effect facts and graph capability copied from the candidate;
7. prerequisites for projection construction, device rebind, value-position
   mapping, descriptors, workspace, and evidence identity.

The bridge may validate a CPE2 prebound projection and populate caller-owned
state before invoking `prepare_candidate`. It may not parse CPE2 in the hot
path, own persistent bytes, query devices, allocate, transfer, synchronize,
reconstruct a conventional format, or hide preparation cost. Unknown projection
kinds, schemas, variants, numeric tuples, and absent bridges fail closed.

Candidate activation remains distinct from planner promotion. The planner must
compare complete measured cost at the requested reuse horizon and retain a
strong conventional fallback.

## Built-in catalog v1

`built_in_candidate_catalog()` returns five immutable descriptors in the table
order above. Each descriptor names the existing factory and registration
function and records the preparation family, numeric family, projection
schema/variant, bounded dense width, caller-owned state size/alignment, output
axis/effect, graph flag, and prebound-view requirements. Transpose explicitly
requires its logical-edge value map and produces the source/feature axis; it
does not reverse the logical relation.

`validate_built_in_candidate_catalog()` checks the summaries against the actual
candidate factories and rejects duplicate identities. The fixed-capacity
`register_built_in_candidate_catalog()` stages all five candidates before
committing them to the existing `candidate_registry`, so duplicate or capacity
failure cannot partially activate the set. Neither operation allocates, owns a
projection, creates runtime state, selects a device, or performs preparation.

Typed preparation remains in the five existing candidate-specific helpers.
CE-LIVE-23 may consume the preparation-family and requirement metadata to route
already-activated projections into those helpers. Catalog presence alone is
never planner promotion and does not hide projection construction, transfer,
descriptor, workspace, value-pack, or synchronization cost.

Checkpoint: `CELLERATOR_BUILTIN_CANDIDATE_CATALOG_V1_READY`.
