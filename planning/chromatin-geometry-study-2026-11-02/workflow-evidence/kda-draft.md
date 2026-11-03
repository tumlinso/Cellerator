# Chromatin-conditioned native CUDA execution study

Status: approved implementation study, 2026-11-02. This directory is a private
study, not an installed API or a trained biological model. The canonical
`cellular_geometry.md` is protected and remains unchanged.

## Contract

Compare supplied Boolean conditions, the existing hard-admission STE, and a
three-input Bernoulli teacher hardened to an eight-bit Boolean table. All
numerical construction, fitting, derivative chaining, reductions and SGD run
in native CUDA. Host code prepares immutable metadata and orchestrates work.

* Supplied: q=b. Numerical gradients are exact for the declared arithmetic;
  there is no gate gradient.
* STE: p=sigmoid(score/T), b=(p>=threshold), q=p*b, with explicitly declared
  dq/dscore=p*(1-p)/T, including every dormant derivative candidate. This is
  a surrogate, not the finite-difference derivative of the hard forward.
* Teacher: row=4*a+2*b+c, eight Bernoulli logits, numerical BCE training,
  and probability>=0.5 hardening. The circuit must equal the thresholded
  teacher. Soft-to-hard discrepancies and hardening costs are separate.

The ordered-product body retains argument order/multiplicity, zeros, shared
parameters and overlapping output incidences. FP16 inputs are multiplied and
accumulated in FP32; outputs and VJPs are FP32. Existing squared-positive-part
gates are regression precedents, not unchanged controls for different q rules.

The linear body preserves the existing GroupedLinear FP16 input, weight,
local output, local dF/dX/dW boundaries and FP32 internal accumulation.
Any fused implementation must round at the same boundaries. Canonical overlap
reduction and dq are FP32. The FP32 alternative is a distinct validation case.

Excluded hard forward branches do not read their participant values or
parameters. STE backward does read declared dormant derivative participants;
its work must not be omitted based on the forward mask. Primary comparisons
use the complete derivative set, without biased dormant sampling.

## Candidates and attribution

1. Direct CUDA scalar/prepared-root-map gates with a warp-uniform numerical
   consumer. Controls are adapted to each exact q and derivative contract.
2. Condition-major planes followed by row bitmap/workset materialization.
3. Direct condition-plane numerical consumption, retaining separate primal
   and derivative membership and avoiding unnecessary bitmap/CSR conversion.
4. Local residual-program cohorts, with cached direct-load and shared-memory
   staging variants. Group inside bounded cell tiles before considering
   global permutation. Raw identities and exact pattern equality survive.

Keep direct-map versus bitplane, materialized versus direct-consumer,
contiguous versus cohort, direct-load versus shared-memory, and rebuild versus
incremental-update ablations distinct. Initial tile candidates are 32/64/128
cells and one/two staging buffers. Target SM70, not later async-copy hardware.

The previous shared-condition study already found that direct maps beat
bitplanes on small DAGs and that workset materialization dominated several
paths. The hypothesis is reduced complete execution cost, not invention of
bitpacking or proof that eliminating divergence alone accelerates the model.

## Workloads and references

An independent C++ scalar oracle and deterministic, versioned fixture manifest
are frozen before optimization. Preserve small seven-node and large 2,304-node
condition cases, coherent/high-diversity programs, sparse/dense activity,
low/high overlap, and frozen/update-every-step parameters. Reuse horizons are
1/16/256 and predicate change fractions 0/1/10/100 percent, using a bounded
named suite rather than a full Cartesian product.

Public replay uses the 10x PBMC no-sorting 10k ARC 1.0.0 dataset. Preserve
original URLs, license, hashes, cell identities and peak coordinates. Reuse
CellShard native ingestion. Observed positive bits are assay observations;
zero observations do not certify biological closure. Neighborhoods and distal
memberships are explicit supplied workload structure, not inferred regulation.
Numerical payloads remain controlled so this is not a biological learning claim.

References: existing regulatory_gates CUDA, shared_conditions experiments,
native gated_product, sparse_support training, and GroupedLinear. Baseplane
hierarchy/conditional operations and KDA KernelBender inform the design;
Baseplane sequence semantics and historical results remain unchanged.

## Validation and measurement

Boolean decisions, identity/order maps and work membership must be bit-exact.
Freeze numerical compound bounds before optimization, retaining the reused
operators' staged-rounding contracts and error bounds. Never relax a validator
to pass a candidate. Include empty/tail cases, arity 32/33, repeated zeros,
shared output/parameter reductions, threshold equality/crossing, stale revisions,
poisoned excluded forward values and nonzero dormant STE gradients.

Condition/schedule invalidation includes accessibility, scores, T, threshold,
teacher/model/mapping revisions and all declared numerical dependencies.
Descriptors cache computation structure, not numerical outputs. Recompute gate
values after parameter changes even when membership does not change.

Run memcheck, racecheck, initcheck and synccheck on representative cases before
measurement. Report cold preparation and recurrent forward/backward/update
cost, memory, conversion, routing, transfer, output recovery and reuse break-even.
Use rotating-order paired samples with independent repeats. Recommend only
correct specializations with >5% practical improvement and a paired 95%
interval supporting improvement over the strongest matched control.

## Commands and resource authority

The standalone CMake target and runner expose CPU reference checks and separate
CUDA correctness/timing entry points. Exact CLI names are owned by the shared
integration contract and recorded in the study README before the first run.
All GPU commands execute from immutable Autobench captures through Device
Scheduler. KDA registers candidate identities and evaluates passing candidates.

The total budget is 14,400 aggregate GPU-seconds (240 GPU-minutes). Use all
available V100s eagerly, one UUID-exclusive device per independent job, with
paired control/candidate measurements on the same device. Every job has a
900-second hard maximum. Count all participating GPU time, including sanitizer
and profiler jobs. Do not hold devices during builds/data preparation.

Host preparation shares eight CPU cores and 16 GiB. Confirm isolated versus
concurrent timing before final measurements and serialize contaminated transfer
phases. Parent Autobench pause and historical paused campaigns are preserved.
The study has its own enrollment and hashed dependency archive inside its
snapshot root; no external path or symlink snapshot dependencies.

Stop submissions at the budget limit and report unfinished required coverage.
A negative complete study is valid; incomplete coverage is not completion.

## Ownership and integration

Root owns provisioning, protocol, resource coordination and final acceptance.
Terra integration owns shared interfaces/build/runner and combined validation.
Principal owns controls/direct consumers/cohorts. Terra engineers independently
own STE and teacher CUDA. Luna execution owns independent references, fixtures
and data preparation. Terra optimization owns measurement interpretation;
independent review and verification follow fixed artifacts. At most eight
agents including root are active. Preserve all pre-existing dirty work.
