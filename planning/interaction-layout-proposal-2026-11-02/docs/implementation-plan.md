# CUDA implementation plan for the factorized interaction layout

Planning only, requested 2026-11-02. Implementation, compilation and GPU
qualification have not started. This operationalizes the
[layout proposal](draft.md); it does not register a campaign or change live Todo
state. Existing paused optimization campaigns remain separate.

## Objective and first complete slice

Deliver a native-first, reusable interaction layout with a complete CUDA
forward/VJP path, optional ordinary Torch autograd integration, and measured
performance against strong equivalent implementations. Start on V100/SM70,
using the installed CUDA 12.9 toolchain. Actual execution must rediscover the
granted device UUID and capabilities; no device availability is assumed here.

The first evaluator is an explicit gated joint law, not the definition of every
Cellerator mechanism:

```text
g[m,c] = sigmoid(bias[m] + sum_k gamma[m,k] * z[c,context[m,k]])
r[m,c,d] = g[m,c] * alpha[m,d] * product_r x[c,participant[m,r],d]
y[c,target[m],d] += r[m,c,d]
```

Initial arities are 2, 3 and 4, with 8 an extension after the first decision.
Roles are ordered; repeated roles and shared/tied parameter bindings are valid.
Several mechanisms can contribute to the same target. Parameters are shared
across cells, and input/output generations are functional. Context is numerical.
There is no threshold pruning of continuous gates.

FP16 is the default payload/parameter storage route. Gate and joint-law
arithmetic, reductions and native gradient buffers use FP32. Output storage
defaults to FP16 with an explicit FP32 option. FP32 payloads are an explicit
alternative. Declare each type independently; respect caller-selected types.
The optional Torch adapter returns gradients in the original tensor dtype;
FP32 native accumulation is not permission to change a parameter's public dtype.
Any packed parameter representation is refreshed after updates and counted.

## 1. Freeze contracts and the independent oracle

- Reuse stable logical occurrences, mechanism IDs and structure epochs from
  [ExplicitPattern](../../../include/Cellerator/geometry/dependency/explicit_pattern.hh).
  Keep physical packing and mutable numerical generations separate.
- Define compact native descriptors, borrowed value views, explicit output/VJP
  buffers, streams, workspace sizes and tape ownership. The core has no Torch
  types, mandatory model superclass, or per-mechanism function dispatch.
- Specify law/shape/precision specialization and explicit unsupported regimes.
  Freeze duplicate-role, empty-support, aliasing, mutation, cross-stream and
  parameter-sharing behavior before optimization.
- Build an independent CPU/Torch reference and high-precision supplementary
  checks. Differentiate products using prefix/suffix products or bounded-arity
  recomputation, never division by an input. Gate gradients reduce contributions
  from all channels before applying the sigmoid derivative.
- Establish and freeze precision-specific forward/VJP tolerances, finite-input
  ranges and exceptional-value policy before candidate search. Include zeros,
  negative values, cancellation, repeated roles, saturated gates, empty cases,
  tied parameters and duplicate outputs. Unsupported ranges fail explicitly.

Deliverable: an operator contract, a trusted oracle and deterministic fixtures
outside candidate-controlled source. Host validation must reconstruct every
logical role and identity from its physical encoding.

## 2. Build a strong parallel CUDA control

Implement a fused direct-gather evaluator of exactly the same law before
introducing packet staging. Assign warps to mechanism/cell/channel work, use
coalesced cell lanes, and parallelize the numerical work. It is not a serial
row evaluator or a Python loop. Build the corresponding complete VJP and
output/parameter reductions.

Keep this evaluator as a measured alternative. Compare the packet candidate
against it with identical arithmetic and thread ownership where possible, and
also compare an optimized Torch/ATen route and applicable library/blocked
implementations. The existing serialized attention kernel is not an adequate
sole performance baseline.

Deliverable: a correctness-qualified, parallel native control and reliable
full-forward/backward benchmark boundary.

## 3. Implement bounded preparation and the physical records

- Bin mechanisms by law, arity, dimensions and precision. Use participant/context
  postings to greedily group overlap and output locality without constructing
  every possible pair of mechanisms.
- Begin with candidate packet limits of 16/32 mechanisms and 64 unique numerical
  input occurrences, with a separate explicit budget for scalar context. Split
  or route oversized cases to the direct evaluator; preserve all mechanisms.
- Emit dictionary entries, role-major 8/16-bit local refs, context refs,
  canonical parameter bindings, output postings, reverse participant postings,
  identity maps and structural guards in bulk allocations.
- Prepare a producer/consumer-compatible numerical order with explicit boundary
  conversions. Stable topology does not imply stable numerical values.
- Defer broadcast/exception metadata compression until measurements show that
  metadata or decoding is consequential. Avoid heap allocation per mechanism.

Deliverable: host preparation, immutable device structure, byte/workspace
accounting and structural round-trip tests. Preparation cost is measured.

## 4. Implement and compare packet kernels

Forward candidates:

- Grid over packet and cell tile. For scalar channels, test a warp over 32 cells;
  for vector channels, test eight cells by four channel lanes per warp.
- Start with 128/256-thread CTAs and channel tiles of 16, then retain only
  meaningful variants. Cooperatively stage unique inputs and reuse them through
  local role refs. All participating lanes obey the declared barrier contract.
- Compute a gate once per mechanism/cell where the ownership/lifetime permits;
  a split-channel launch must account for repeated gate work or saved gates.
- Use local output postings to combine duplicate destinations. Emit unique
  packet/output partials, followed by a grouped cross-packet reduction. Compare
  an atomic variant only under an explicit numerical/determinism contract.

Backward candidates:

- Use participant-owned inverse postings for the first deterministic route,
  summing every incident role and bounded-arity recomputation. Preserve repeated
  roles and accumulate tied-parameter gradients by canonical parameter identity.
- Give backward its own cell/channel tile and shorter live ranges. Staging
  64 FP16 inputs for eight cells by 16 channels is 16 KiB; FP32 gradients for
  the same dictionary require another 32 KiB before any other scratch. Do not
  assume that the forward tile fits backward.
- Compare bounded recomputation with saved intermediates, including tape memory
  and mutation/lifetime restrictions. Reduce parameter/context partials across
  channels and cells explicitly.

The product evaluator is initially a fused SIMT/reuse experiment. Dense learned
projections or a later joint neural law must retain cuBLAS/CUTLASS controls and
may justify Tensor Cores. Do not force the product into matrix arithmetic or
assume Ampere-only features on SM70. No PTX/SASS work is implied by this plan.

Deliverable: complete packet forward/VJP candidates with bounded scratch,
correct streams, immutable structure and retained live input/parameter owners.

## 5. Qualify performance and select the useful regimes

The first discriminating sweep fixes 32 four-input mechanisms per candidate
packet and varies distinct participant count from 16 through 128. Larger
dictionaries split or use the supported direct route. Hold parameters, output
contention, arithmetic and scheduling choices fixed. Test shared and independent
conditioning and both scalar/modest vector channels. This distinguishes input
reuse from unrelated kernel improvements.

Then cover low-overlap/random negatives, shared-core mechanisms, high output
fan-in, varied widths/batches, all-changing values, parameter updates, topology
churn and boundary conversion. Include a supplied real candidate graph when its
dataset/learning question is selected; synthetic success is not biological proof.

Use the installed [Autobench skill](../../../../autobench/skills/autobench/SKILL.md)
and canonical `autobench` route for registration, timing, profiling and registered
evidence. Register only this new scope when execution is authorized. Capture
narrow deterministic source/data closures and actual binaries in managed
`{build}` artifacts. Use KDA candidate/evaluation records for repeated comparison.

Required evidence before promotion:

- Forward/VJP validation and applicable memcheck, racecheck, initcheck and
  synccheck pass on the exact captured executable. Require zero unexplained
  skips, missing cases and sanitizer findings.
- Pair unprofiled measurements against the strongest valid baseline with a
  predeclared case corpus, warmups, repetitions, confidence criterion and
  minimum meaningful improvement. Freeze these before measurement.
- Report preparation, transfers/conversion, forward, backward, output/gradient
  reduction, parameter refresh, live memory and reuse horizon. Also report the
  complete training step and a steady-state view separately.
- Profile only consequential losses identified by ordinary timing: timeline
  first, then memory traffic, bank conflicts, register spills, occupancy or
  reduction bottlenecks as appropriate. Profiler replay time is diagnostic.
- Select the packet route only for supported, demonstrated winning regimes;
  include dispatch and fallback costs in confirmation. Reject it for regimes
  where the complete recurring cost loses. High-overlap losses are a reason to
  inspect the basic reuse premise before adding sophisticated compression.

All device work is finite, UUID-bound, Scheduler-admitted and protected against
measurement interference. Establish the actual resource/time envelope in the
execution task before registration; initial runs use one GPU at a time. Preserve
source/binary identities, failed candidates and verified cleanup receipts.
This plan submits no device job and does not restore paused registry entries.

## 6. Integrate the winning native path and demonstrate training

Proposed ownership, subject to current source-owner reconciliation:

| Responsibility | Existing owner and likely additive route |
| --- | --- |
| Host physical descriptors/preparation | `include/Cellerator/layout/`, `src/layout/`; consume semantic patterns rather than redefine them. |
| Native evaluator and explicit VJP | `include/Cellerator/modeling/cuda/`, `src/modeling/cuda/`; build through `cmake/CelleratorFrameworkCUDA.cmake`. |
| Buffers, events, streams | [existing memory owner](../../../include/Cellerator/memory/cuda/resources.hh); no second resource authority. |
| Optional ATen/autograd adaptation | `src/nn/ops_cuda.cc`, `bindings/nn/`, thin Python adapter; retain original owners/current stream and first-order boundary. |
| Qualification and measurement | Focused host/CUDA tests, trusted scripts and benchmark harness under this task's workspace. |

Keep the generic native CUDA build independent of the NN/Torch build. Existing
`Cellerator::nn_cuda` requires CUDA-enabled Torch and cannot be the native core
dependency. CUDA NN CTest entries currently provide disabled runtime tests;
compile/link success alone is not device qualification. GPU tests run through
admission, not ordinary unattended CTest.

Before implementation, add the required custom-op registry entry for the optional
Torch op, define reproducible build/validate/evaluate commands and narrow source
manifests, and bind them to the task harness. Commands/targets do not exist yet;
they must not be presented as checks already run. Native and installed-consumer
builds, ordinary optimizer composition, repeated outstanding tapes, mutation
rejection, non-default streams and cross-stream allocator lifetimes are required.

Finish with a small C++ authored model and optional Python training example
using shared parameters, real numerical context and repeated optimizer updates.
Record actual complete-step results and supported regimes. Attention with exact
cross-packet softmax/VJP and more general device-callable laws are follow-on
extensions after the first slice establishes a worthwhile execution contract.

## Parallel ownership and completion

Root retains model direction, public contracts, integration and final acceptance.
A Terra engineering owner can carry coupled CUDA design/implementation/validation;
fresh Luna assignments can independently own the oracle, established host
preparation, harness and bounded candidate checks once their contracts are clear.
Keep overlapping source edits attributable and reconcile current active work.

Completion requires a native usable implementation, correct complete VJP,
optional adapter with ordinary optimizer behavior, installed/build evidence,
source-bound device qualification and an honest regime-specific performance
decision. A correct rejection of the new layout preserves a useful direct
evaluator and evidence; it is not a basis for claiming the layout is faster.

Planning basis: current proposal, current CMake/source owners, installed tool
locations, Project Control source/toolchain reads, CUDA/KernelBender/KernelWiki
guidance and Autobench's installed skill. No new speedup or globally optimal
implementation claim is made.
