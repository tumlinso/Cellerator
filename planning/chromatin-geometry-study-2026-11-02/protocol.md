# Reviewed execution protocol

The approved implementation contract is
[the study draft](../../bench/chromatin_geometry/docs/draft.md). The user-approved
[workflow exception](workflow-exception.md) permits implementation while native
verification is repaired. Neither document is a measured result.

## Comparisons

Compare execution arms only for identical gate semantics, numerical precision,
ordered participant/output mappings, derivative membership, and update policy.
Body-only measurements with precomputed gates are diagnostic and must be labelled
separately from condition-to-update measurements. The smooth product gate is a
regression reference for its own semantics.

Forward exclusion permits no reads of excluded source participants. In the STE
route, backward evaluates all declared derivative candidates, including dormant
ones; any dormant local values required for gate derivatives are computed in
backward. A staging buffer containing an explicitly constructed zero is distinct
from reading an excluded source value. Count packing and any additional passes.

Predicate-change cases modify input observations and propagate their changes
through the actual condition DAG. A changed root bitmap alone does not establish
dependency-driven evaluation. Saved descriptors identify structure, membership,
and revisions; they never establish that numerical values remain current.

## Evidence gates

1. Freeze deterministic workload definitions, independent references, and
   numerical bounds before performance tuning. Retain their source hashes.
2. Establish exact Boolean decisions, tail bits, identities, work membership,
   and guard behavior. Validate outputs, VJPs, reductions, and parameter updates
   against independent references. Validate STE against its declared estimator.
3. Establish combined producer → body → backward → update correctness for all
   three gate routes and both numerical bodies. Isolated component tests do not
   establish this combined behavior. Failed numerical construction must not
   silently publish a successful training step.
4. Run representative memory, race, initialization, and synchronization checks
   under Scheduler admission. Bind each result to source, binary, input,
   toolchain, device UUID, job, and verified cleanup.
5. Measure cold and recurring costs, including applicable preparation,
   conversion, fitting, hardening, transfer, routing, packing, update, and output
   recovery. Report body-only and resident kernel times as bounded components.
6. Rotate candidate/control order within independent paired trials on the same
   device. Check concurrent versus isolated timing. Profiler replay times are
   diagnostic; they are not confirmation timing samples.
7. Recommend a specialization only when it passes correctness, improves the
   complete matched comparison by more than 5%, and its paired 95% interval
   supports improvement over the strongest applicable control. Retain rejected
   and inconclusive candidates and unfavorable regimes.

## Resource accounting and reproducibility

The study has one aggregate limit of 14,400 GPU-seconds and a 900-second maximum
per job. Reserve a bounded attempt before any submission-capable controller
action. Failed and instrumented jobs count. Ambiguous execution or unverified
cleanup retains its reservation. CPU-only preparation does not reserve a GPU.
Study worker enrollments share the aggregate ledger and preserve the parent
workspace's pause.

Immutable captures contain the actual reused source dependencies, including the
CellShard checkout used by native ingestion and the KDA evaluation engine. A
different checkout with the same library name is not the same source closure.
Archive members must be regular files with no escaping paths or symlinks.

The public input is the
[10x PBMC Multiome ARC 1.0.0 dataset](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-no-cell-sorting-10-k-1-standard-1-0-0).
The downloaded filtered feature-barcode HDF5 file has SHA-256
`0668e2bf9ff521a8e5e7aa37717f3780bad84081f26f888df16a9398d5020e76`.
Native conversion preserves cell/peak identities and coordinates, records its
own hashes, and uses positive observations as replay bits. Observed zeros are
not certified biological exclusions. Neighborhoods and numerical payloads are
declared fixtures; this study does not discover regulatory biology.

Required coverage that is missing, failed, or inconclusive remains explicit.
At the budget limit, stop submissions and report unfinished requirements.
