# Prepared entity-to-mechanism union

## Task contract

- **Task:** `prepared_entity_mechanism_routing`
- **Objective:** replace the per-call CUDA-input-to-Python-list-to-CUDA-output
  detour in `SparseSupportModule.affected_mechanism_workset` with an opt-in,
  prepared CUDA entity-to-mechanism posting-union query.
- **Current baseline identity:** shared workspace Git HEAD
  `1d1dfda8c5b5ace00415c9bd0847f975f6b655b4`; baseline adapter file SHA-256
  `adcaca5c59a239349f4454928f9a9fb441e0f6e20b7031be8cc02af4387d9fbf`
  (`python/cellerator/torch/execution.py`). Native query source identity:
  `src/indexing/cuda_postings.cu` SHA-256
  `d034ac6815e070b356d08b3f8852cf03d7ab39a91d008ab12008bbf30c92f99c`.
- **Target:** Tesla V100, CUDA `sm_70`; build only for `sm_70` during this
  qualification. No live device identity or resource grant is assumed; the root
  owns Device Scheduler admission.
- **Input:** contiguous CUDA `int32` or `int64` entity ordinal tensor
  `[cells, participants]`. Negative ordinals are padding and ignored; any
  nonnegative ordinal outside the prepared entity domain is an error. Existing
  route table rows are the source of truth: each entity maps to its current
  sorted zero-based mechanism indices, with at most two routes per entity under
  the existing route policy.
- **Output:** device CSR `(offsets:int64[cells+1], mechanisms:int32[n])`.
  Within each cell, mechanism indices are sorted, unique, and equal to the exact
  global union of postings for that cell's nonnegative input entities. Empty
  rows, duplicate entities, unsorted entities, and repeated route memberships
  preserve current behavior. No parameter tensor is created or owned by the
  helper.
- **Owner and invalidation:** prepared object binds an explicit route-owner
  identity, model structure epoch, route/value generation, and device binding
  generation. A change to any bound key rejects or replaces the object. It
  consumes the package's already computed `_mechanism_route_postings`; it does
  not score or reinterpret routes. Runtime scratch is per stream and bounded;
  output tensors remain per-call and immutable for autograd tape lifetime.
- **Module API:** `module.affected_mechanism_workset(entity_ids)` keeps the
  current Python implementation as its default. Callers can opt in with
  `strategy="prepared"`; route scoring and the route table remain owned by the
  existing module. The owner key is the module object identity, the relation
  keys come from the current base support snapshot, the epoch is
  `module.structure_epoch`, the route generation is the native published
  parameter generation, and the binding generation advances when the fixed
  CUDA device, parameter pointer, or native handle binding changes. Activity
  changes supply new entity IDs and reuse the index. Value/route generation,
  support publication/restore, and binding changes rebuild the projection.
  `application_state` records prepared-index count and preparation seconds,
  prepared query count, and the number of explicit count synchronizations.
- **Zero-valued generation:** a native published route generation of zero is
  valid. Owner/relation/structure/epoch/binding identities must remain positive;
  no offset or synthetic generation is substituted.
- **Capacity and failure:** index and query scratch/output capacities are
  caller-derived and checked with overflow-safe arithmetic. The complete
  posting query reports exact required output count. If exact output size
  requires reading the device status on host, use one explicit current-stream
  synchronization before allocating the exact mechanism tensor; surface that
  sync in timing. Query overflow or invalid IDs fail before returning a partial
  workset.
- **Independent oracle:** the incumbent Python semantics,
  `sorted({m for e in row if e >= 0 for m in route_postings[e]})`, computed from
  frozen inputs and the exact same route table. Compare byte-for-byte offsets
  and mechanism indices on CPU and admitted GPU runs.
- **Baseline:** current `affected_mechanism_workset`: `.cpu().tolist()`, Python
  per-cell `set` union, and upload of exact result tensors. The score refresh
  that produced the route table is shared setup and remains unchanged.
- **Strong Torch alternative:** a cold dense boolean `[entities, mechanisms]`
  route matrix followed by device gather, participant reduction, row counts,
  prefix scan, and `nonzero` materialization. Include matrix construction and
  upload in lifecycle cost. This checks whether the host bridge is worth
  replacing when the mechanism axis is small.
- **Candidate:** build a native canonical reverse-posting index once from the
  current route table; upload the existing `cuda_posting_projection_v1`; on
  each query adapt entity ordinals to role/entity terms on the current stream
  and call existing `cuda_query_reverse_postings_any_v1`. Do not add a new
  planner or a second route-scoring implementation.
- **Representative workload:** existing synthetic consumer uses ranked input
  `[32,8]` and event input `[32,12]` (`python/cellerator/torch/testing.py`;
  training example). Include stress shapes `[256,64]` and `[1024,128]`, with
  entities `4096`, mechanisms `16`, and the existing maximum route degree of 2.
  Synthetic data is plumbing/performance evidence only, not a biological claim.
- **Evaluation horizons:** cold index construction plus upload, one warm query
  (ranked forward horizon 1), event-sequence horizons 12 and 16, and 64 repeated
  queries as an explicit reuse sensitivity. Time full cold preparation and
  complete warm calls. The prepared call includes query conversion, posting
  kernels, count/status synchronization, final-ID emission, allocator activity,
  and returned CSR tensors. Substage times are not isolated because doing so
  would instrument the hot path; no kernel-only timing is used for a decision.
  Keep both baselines end-to-end on the same fixed inputs, including current
  input materialization and output upload. Compute paired lifecycle improvements
  at each horizon and bootstrap a 95% interval from 21 paired samples.
- **Success target:** exact semantics and no sanitizer failures are mandatory.
  Keep the candidate opt-in. Claim a useful regime only if paired complete-path
  timings show a positive improvement with 95% confidence for that regime after
  cold setup amortization. Do not promote a default based on isolated kernels or
  query-only timing. If horizon 1 loses but longer horizons win, preserve that
  boundary explicitly; otherwise retain it as a correctness prototype or reject
  the performance lead.
- **Allowed:** additive files under `include/Cellerator/ops/sparse_support`,
  `src/ops/sparse_support`, `bindings/sparse_support`,
  `python/cellerator/torch`, `tests/prepared_routing`, and this experiment's
  benchmark/evidence directory, plus the module's route-cache and
  `affected_mechanism_workset` strategy integration in `execution.py`.
  CMake and `module.cc` registration remain root-owned.
- **Not authorized:** GPU execution before explicit root admission, GPU probes,
  edits to existing `training.*`, `module.cc`, `routing.py`, CMake files, Git
  refs, Todo state, or transition-mode semantics.

## Design choice and falsifier

The incumbent sparse representation already has the needed exact operation:
reverse postings by `(role, entity_id)` and the CUDA any-of query emits a sorted
unique union of logical IDs. Encode a route table as one logical edge per
mechanism, with each entity as a role-zero participant; use stable edge ID
`mechanism_index + 1`, then emit `logical_edge_id - 1`. This preserves the
current sorted set-union contract, including duplicate and unsorted queries,
while allowing skipped empty mechanism postings.

The first candidate is the existing prepared reverse-posting projection and
query, plus only the adapter kernel needed to turn padded entity ordinals into
query terms and emit exact mechanism indices. A dense route matrix is measured
only as a Torch baseline, not added as a prepared product candidate; a new
segmented-union kernel is also outside this probe. If query setup, scratch
traffic, or the mandatory count synchronization erases the complete path
benefit at real horizons, that is the falsifier; keep the Python route as the
fallback.

## Executable checks

1. Host builder tests: duplicate/unsorted membership rows, empty entity rows,
   skipped mechanisms, invalid mechanism IDs, invalid identities, capacity
   bounds, and exact host-query comparison with the independent oracle.
2. Static CUDA compile for `sm_70`; no device use during this gate.
3. After root admission, a scheduler correctness job compares exact outputs on
   the two repository shapes and both stress shapes, including all-negative,
   duplicate, unsorted, empty-participant, and out-of-range cases. Run
   Compute-Sanitizer memcheck, racecheck, and synccheck as separate correctness
   stages with source/binary hashes.
4. After correctness, one finite exclusive V100 job performs paired complete
   path timings for incumbent Python, dense Torch, and prepared postings at
   horizons `1,12,16,64`, using identical fixed route tables and entity inputs,
   5 warmups and 21 paired samples. Measure cold preparation separately and
   calculate full lifecycle cost and paired 95% intervals at each horizon. Do
   not overlap benchmark and sanitizer samples.

## Local implementation evidence

The host builder and Torch binding are new files in the assigned worktree
directory. The C++ CPU test was manually built and passed, as did the Python
adapter contract tests. `prepared_routing.cu` compiled offline for `sm_70`; the
Torch binding translation unit compiled against the selected Torch 2.7.0+cu126
headers. These checks do not execute CUDA and do not establish runtime safety or
performance.

After root-owned CMake integration, configure a fresh build with hardware
probing disabled and build `cellerator_python_torch` for `sm_70`. The root must
resolve the exact `_torch*.so` target file from that build. With that binary
path substituted as `TORCH_MODULE`, the admitted correctness command is
`python tests/prepared_routing/run_cuda_correctness.py --device-index 0
--torch-module-file "$TORCH_MODULE"`. Run that command separately under
Compute-Sanitizer memcheck, racecheck, and synccheck. The admitted benchmark
command is `python bench/sparse_support/experiments/prepared_routing/run_prepared_routing.py
--device-index 0 --torch-module-file "$TORCH_MODULE" --warmups 5 --repeats 21
--output-json /tmp/cellerator-prepared-routing-results/paired-route-paths.json`.

Exact configure argv, with no configure-time device probe: cmake -S
/tmp/cellerator-coherence -B
/tmp/cellerator-coherence-prepared-routing-cmake -DCELLERATOR_ENABLE_CUDA=ON
-DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF
-DCELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES=OFF -DCMAKE_CUDA_ARCHITECTURES=70
-DCELLERATOR_ENABLE_TORCH=ON -DCELLERATOR_ENABLE_PYTHON=ON
-DCELLERATOR_BUILD_TESTS=OFF. Build argv: cmake --build
/tmp/cellerator-coherence-prepared-routing-cmake --target
cellerator_python_torch -j 4.

GPU UUID, binary SHA-256, full source SHA-256 manifest, exact scheduler job IDs,
and resulting timing/correctness receipts remain pending root-owned admission.
Proposed budget: one 15-minute finite exclusive job for correctness and
separate finite 15-minute jobs for each sanitizer stage plus one 20-minute
timing job, `max_attempts=1`; the root decides whether to reduce or split it.
No job is submitted by this task.
