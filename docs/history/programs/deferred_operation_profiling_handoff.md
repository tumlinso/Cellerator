# CE-EXOP deferred profiling handoff

## Status and evidence anchor

CE-EXOP is profiler-ready, not performance-promoted. The completed acceptance
surface provides deterministic synthetic fixtures, stable candidate and stage
identities, profiler markers, generic partition export, resource receipts, and
sm_70 compile checks. It does not provide a deep profile, a biological-data
result, a candidate winner, or a preprint claim.

The authoritative evidence record is
[`bench/ce_exop/evidence/286/documentation_spine_v1.json`](../bench/ce_exop/evidence/286/documentation_spine_v1.json),
SHA-256
`dceef389e921cad8f9bef1521eee1b83a5f1b04b810bda69bb0eed1c921fd3f3`.
It was published on main `72b7ae68` and pins the accepted implementation commit
`eef1fd25013a38eaf2034b164760ba0f63dd5ded` and tree
`c71f1308e310c94a96c5b8807f50b9c9cc782096`.

CE-EXOP-282 reached `CE-EXOP-PROFILER-READY` at authority revision 3588.
Its ordered profiler-artifact manifest SHA-256 is
`5cde357d609425e51b779bc19533b8108171ca1c024bc3714b544b07ac51aa5c`.
That acceptance explicitly recorded `compile_only=true` and
`cuda_runtime_executed=false`.

Separately, CE-EXOP-255 completed a bounded, claim-bound V100 correctness run.
It used lease `de3474b4-6300-4a51-99ee-f354c281198c` for only the integrated
CUDA runtime acceptance matrix and two Compute Sanitizer memcheck targets. The
declared matrix passed and memcheck reported zero errors. That evidence proves
correctness for its exact runner, device, and toolchain; it is not timing or
promotion evidence.

## Work that remains deferred

The following campaigns have not been performed:

- deep Nsight Systems or Nsight Compute profiling;
- end-to-end biological-data validation;
- cost-surface calibration or candidate promotion;
- preprint figures, ablations, evidence, or claims;
- CE-AMP work.

No result from the synthetic fixtures may be described as measured hardware
performance. No CE-EXOP candidate is a universal winner. Experimental
candidates retain their recorded dispositions until a separately authorized
campaign supplies comparable end-to-end evidence.

## Authorization and prerequisites

Before any deferred campaign starts:

1. Project Control must expose a task that explicitly authorizes the campaign
   and its repository paths.
2. Deep profiling requires `CE-EXOP-DEEP-PROFILING=authorized`.
3. Every GPU command requires a claim-bound `accelerator:any` lease. Comparable
   timing also requires the repository `cuda-benchmark-mutex` for the entire
   measured interval.
4. The source commit, tree, build configuration, fixture or dataset identity,
   and generated binary identity must be frozen before measurement.
5. Exact output correctness must pass before throughput is recorded. New
   device formats or pointer rebinding must also pass the applicable Compute
   Sanitizer checks.
6. Setup, projection construction, packing, transfers, synchronization,
   epilogue, and output-order transforms must be classified as included,
   excluded, or amortized before candidates are compared.

Do not reuse the CE-EXOP-255 lease. It was released automatically when that
task completed. Its result may be cited only with its exact evidence identity.

## Required campaign record

Every hardware or profiling result must record all of the following:

- Project Control task, claim, request, lease, resource instance, lease start
  and release times, and benchmark-mutex receipt when timing;
- GPU product, full UUID, compute capability, memory size, power/performance
  mode, clock policy, ECC state, MIG state where applicable, and topology;
- host CPU, NUMA placement, kernel, driver, CUDA toolkit, nvcc, host compiler,
  CMake, relevant libraries, profiler, and Compute Sanitizer versions;
- Git commit and tree, dirty-state fingerprint, build type, CUDA architecture,
  compiler flags, fast-math state, and binary or cubin hash;
- logical domains and orders, structure and value identities, value generation,
  shape, dtype, accumulation policy, degree/width/segment distributions,
  residual fraction, value mode, fusion choice, and work-window grouping;
- warmup and repeat counts, stream and concurrency policy, graph-capture state,
  workspace, preparation/reuse assumptions, and every included or excluded
  cost;
- exact reference, numerical policy, tolerance, mismatch count, maximum error,
  sanitizer result, and fused/unfused equivalence result;
- latency distribution, throughput, bytes per useful interaction, useful
  interactions per DRAM byte, launch count, host time, preparation cost,
  persistent metadata, transient workspace, and memory expansion;
- when captured, achieved bandwidth, registers, shared memory, occupancy, warp
  and branch efficiency, L1/L2 behavior, stalls, atomics, and synchronization;
- the strongest applicable baseline and the reason every omitted baseline is
  inapplicable.

Report the dominant expected limiter explicitly: HBM traffic, L2/shared-memory
reuse, register pressure, occupancy, launch overhead, PCIe/NVLink, atomics,
synchronization, or host preparation.

## Profiler identities and receipts

Use the frozen files under `bench/ce_exop/` as inputs, not as measured output:

- `profiler_candidate_matrix.tsv` defines candidate identities;
- `profiler_fixture_matrix.tsv` defines controlled synthetic regimes;
- `profiler_stage_manifest.tsv` defines source-correlated stages;
- `profiling_marker_manifest.tsv` defines profiler markers;
- `profiler_resource_receipts.tsv` proves that fixture generation made no
  device query or GPU lease request;
- `generic_partition_export_v1.json` carries portable partition identities;
- `profiler_readiness_acceptance_v1.json` records readiness acceptance; and
- `deferred_profiling_requirements_v1.json` records the authorization boundary.

Verify these identities and hashes before capture. A measured record must link
its samples back to candidate, stage, marker, fixture or dataset, source, build,
and resource-receipt identities. Missing or ambiguous correlation invalidates
the capture; it must not be repaired by relabeling samples after the run.

## Biological-data provenance

Biological validation is a separate authorized campaign. Record dataset name,
provider, accession and version, retrieval time, license, checksum, exact raw
files, filtering and QC policy, feature and cell identifiers, order maps,
normalization and transformation policy, split or sampling seed, structural
and value identities, generated fixtures, and every conversion. Keep source
data and derived manifests outside Cellerator ownership where required; the
repository must not become the storage owner.

Correctness must be established against an independently specified reference
before biological throughput is compared. Report selection bias, excluded
samples, failed runs, representativeness limits, and whether the workload was
chosen before or after observing performance.

## Promotion and preprint boundary

Candidate promotion requires reproducible end-to-end evidence on the strongest
relevant alternatives, including preparation and transformation costs under a
stated reuse model. A microkernel win, one synthetic regime, or one V100 run is
insufficient. Preserve pure sparse fallback, exact support, logical edge
identity, projection plurality, and planner authority when evaluating a path.

Preprint work begins only under its own authorization after the measurement
record is frozen and independently reviewed. Figures must trace to immutable
data and scripts, show uncertainty and negative results, and distinguish
mechanism controls from biological evidence. Do not convert profiler readiness
into a performance, biological, or publication-readiness claim.

## Stop conditions

Stop the campaign and preserve the partial record if any of these occurs:

- the Project Control claim, accelerator lease, or benchmark mutex is absent,
  expired, released, or does not cover the command;
- the source tree, binary, dataset, topology, power/clock state, or toolchain
  changes after the campaign identity is frozen;
- exact correctness, fused/unfused equivalence, sanitizer, race/init, or
  numerical-policy checks fail;
- stage, marker, candidate, dataset, or resource-receipt correlation is missing;
- setup or transformation costs cannot be classified consistently across
  candidates;
- thermal, clock, contention, background activity, or topology contamination
  makes repeats incomparable;
- a result would require an undeclared dataset, deep-profile scope, promotion,
  preprint claim, distributed orchestration, sm_86 work, or CE-AMP work; or
- evidence suggests a new architecture or ownership decision rather than a
  bounded measurement.

On stop, release resources through Project Control, label the evidence invalid
or incomplete, retain diagnostics, and request a new scoped task. Do not widen
the campaign locally.

## CE-AMP interlock

`CE-AMP-PERMISSION` remains `not_granted`. CE-EXOP profiler readiness and even
future CE-EXOP completion do not authorize CE-AMP. Every CE-AMP task continues
to require both `CE-EXOP-COMPLETE` and an explicit
`CE-AMP-PERMISSION=granted` decision. This handoff neither reaches that
checkpoint nor changes that decision.
