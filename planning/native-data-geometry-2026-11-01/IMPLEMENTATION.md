# Native cellular-geometry data cutover

Implementation record, 2026-11-02. This records the realized source boundaries
and their qualification, separately from the original reviewed proposal.
The [implemented type map](implemented-type-map.csv) reconciles all 71 planned
entries with actual symbols, headers and deliberate retained owners. The
[native guide](../../docs/native_data.qmd) is the current user entry point.
The [numerical/execution contract](IMPLEMENTATION_CONTRACT.md) fixes the
operation, precision, workloads, tolerances and resource policy.

## Realized boundaries

Persistent identity and logical occurrences live in `structure`. Physical axis,
occurrence, local-index and projection maps live in `layout`. Biological labels
and annotations remain in `biology`; `execution::FeatureSourceBinding` owns the
actual prepared alignment. Domain-local entity/occurrence zero is valid;
persistent IDs, runtime handles, batch ordinals and physical slots are distinct.

Dependency foundations live in `geometry::dependency`, including explicit and
implicit participant families, applicability and knowledge scope, source and
selection associations, and restricted condition inputs. Generic posting
storage/query contracts live in `containers`; semantic posting construction
remains beside the dependency grammar. A retained pattern owns its participant
postings once. Prepared execution borrows them and can explicitly materialize
compact device ordinals. Different condition-read indexes are not duplicates.

Boolean condition computation lives in `ops::condition`. Operation access,
effects, derivative requests and exact omission contracts live in `ops`.
`execution` owns current work, source/time revisions, cached outcomes, retained
content association, readiness and actual realization. A small realized value
never substitutes for an operation-specific omission proof.

Specialist encodings live in `tensor::{block_masked,quantized,state_codes}`.
Views are typed borrowing contracts; bulk owners, immutable metadata and
retained numerical contents remain distinct. CUDA resources use one canonical
`memory::cuda` allocation implementation, with a shared-retention adapter.
Ordinary dense/CSR/COO numerical storage remains caller/library owned. Existing
archive payloads are isolated under `interop::formats`, with minimal CellShard
alias/import updates. No ordinary tensor or autograd replacement was added.

Indexed-mechanism numerical kernels, parameter lifecycle and prepared tape
orchestration were split into reusable translation units. Operation-specific
tapes remain with their numerical family; global realization and program
bindings remain in execution. Namespaces do not create separate libraries for
every responsibility. Existing compiled support/indexing/gating owners remain
where useful, with canonical source/header ownership and checked exports.

CellPack packing plans and CPK1/CPE2 readers retain their existing byte meaning.
Explicit record/tile bridges expose native views without decoding replacement
values. Packing occupancy summaries now live in layout, separate from execution
bindings and biological dependency knowledge. No gratuitous format rewrite was
needed. Full physical-axis/no-axis native manifests retain descriptor v1;
logical-only or zero-extent axes use explicit native v2. The current Python
reader rejects v2 clearly; its existing v1 facade remains supported.

## Connected operation

The bounded example evaluates ordered gated joint products, repeated
participants, overlapping outputs and shared parameters across two devices.
Expression/output/parameter homes are logical device slot 0; regulator home is
slot 1. Device ordinals are supplied explicitly. Canonical row IDs map to batch
ordinals, and physical orders/realization IDs are explicit.

Both devices consume the packed representation directly. The prepared
operation associates semantic demands with actual atom requirements, physical
coverage, external planes, retained values, readiness and existing program-v2
callbacks. It adds no independent graph or population scheduler. First-order
VJPs return to their owners; a checked canonical update precedes replica
publication and the next forward. Zero-rate parameter fields preserve bits.

Inactive participant values are not read. A wholly inactive remote projection
omits its participant transfer. Mixed activity currently transfers the full
prepared projection, including zero placeholders. The distributed adapter
requires output-producing groups and nonempty participant coverage per device;
unsupported compositions fail at preparation. Host/local primitives have their
own broader contracts. These boundaries do not establish general distributed
autograd or biological validity.

## Qualification

Evidence is source/binary bound. Host execution, CUDA compilation, actual CUDA
execution, numerical correctness, sanitizer results and timing are different
claims. The final qualification artifacts below identify their actual scope;
failed admission and startup attempts do not count as numerical checks.

- Backend-disabled native-data suite: 12/12 passed.
- CUDA-configured host-only native-data/support selection: 16/16 passed.
- Independent native Cell/NN CPU suite: 23/23 passed, including no-RTTI core,
  optional bridge, occurrences, feature alignment, manifests and queries.
- Independent installed C++20 module consumer: 1/1 passed with Clang/scanner 18;
  it runs CPU forward/backward/SGD and exchanges ordinary header/module types.
- Installed host/native-CUDA-aware and CellPack record/tile consumers passed;
  these consumer runs make no GPU calls.
- Bounded projection differential/adversarial probes passed after API-spelling
  migration. Original frozen validator source remains retrievable in Git;
  historical result records and their source identities were not rewritten.
- Actual local packed FP16/FP32 and memory/state-code owner GPU checks passed
  with verified Scheduler cleanup.
- The native ATen C++ active/inactive reference campaign passed on the selected
  pair through Autobench. Independent replay verified all eight fields,
  immutable source/binary identities and logical device order. The inactive
  case preserves its nonzero baseline.

The required connected numerical operation passed on two actual admitted
V100 devices. All six frozen executables also passed Compute Sanitizer
memcheck with full leak checking and synccheck: zero reported memory/leak or
synchronization errors, with verified cleanup. Only the explicitly handled
`cudaGetLastError` status 704 in the prepared-transfer constructor is suppressed
by an exact API/error-code/call-chain XML rule. Other API and memory checks
remain enabled. The original unsuppressed/tool-setup attempts are retained;
they are not recorded as clean checks.

| Qualified execution | Scheduler job | Evidence |
|---|---|---|
| Six-program connected/local/owner correctness | `35bc708e-4a68-4471-9583-41c523dcbe73` | [receipt](evidence/gpu/correctness-job.json) |
| Same binaries, memcheck and full leak check | `580c2d9f-d43c-4a2e-be9d-f28d3f96aaf3` | [receipt](evidence/gpu/memcheck-job.json) |
| Same binaries, synccheck | `2d7e1657-ff87-4c70-9b1c-b85c4abf635b` | [receipt](evidence/gpu/synccheck-job.json) |
| Native ATen exact active/inactive oracle | `466bd887-f112-4286-bc04-10a98ac97c8e` | [receipt](evidence/aten/reference-job.json) |
| Paired low/high scale correctness | `ff44d06c-86db-45c6-8143-41779469d9f9` | [receipt](evidence/scale/correctness-job.json), [campaign](evidence/scale/campaign.json) |

Independent regeneration checked the 512-group disjoint workload (51 active,
9.96094%) and the 1,024-group overlapping workload (922 active, 90.03906%).
Each group has 32 participants. Both implementations match independent scalar
outputs, all five VJPs, FP32 theta update and next outputs at the fixed
`atol=rtol=2e-5`. Maximum candidate-to-ATen absolute difference was `1.49e-8`;
maximum independent-formula difference was `2.72e-8`. Logical owner slots and
Scheduler-visible ordinals are verified separately.

The [evidence manifest](evidence/manifest.json) identifies receipts, host logs,
653 build-dependency source hashes, six frozen binaries and lossless compressed
raw scale outputs. Dependency closure is build provenance, not blanket runtime
qualification of every linked subsystem.

## Performance boundary

No timing result or speedup is claimed. All three bounded timing attempts were
refused by the Scheduler's strict 0-percent/two-sample idle precondition before
worker execution. They observed no processes and ample free memory, but
intermittent 3-percent utilization. Numerical tolerances and timing admission
were not relaxed. Their [receipts](evidence/timing/attempts.json) remain with the
qualification. The Autobench workspace was restored to paused with no active
attempts; the selected scale correctness packet remains current.

The retained timing protocol compares synchronized full cycles and backend
setup plus one cold cycle. It includes forward/backward, reductions, update,
replica refresh and initial required source movement. It excludes process
startup and synthetic workload generation. Stage medians are not comparable
one-to-one. Candidate allocator peak is unavailable; prepared allocation
capacity must not be presented as peak usage. The frozen ATen cycle includes
allocator-peak resets that the candidate does not perform; this instrumentation
asymmetry must be accounted for before a later performance promotion.

The functional multi-device/data cutover is qualified within these bounds.
Performance promotion remains open pending a valid measured comparison.

## Preservation

`cellular_geometry.md` remains byte-identical, SHA-256
`d37f504e5c1ab629334f4127dcaa60d008afe94e560d41aa936d8ba161124254`.
The pre-existing Cell/model work was preserved before coordinated changes.
No workflow database, generated Todo projection, old benchmark result or
unknown Scheduler lease was hand-edited. The user explicitly waived the task
handle requirement; physical GPU admission and evidence requirements remain.
