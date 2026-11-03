# A factorized layout for conditional joint interactions

Disposition: **propose a bounded design probe**. This document proposes a new
Cellerator execution representation. It does not implement it, select a public
API, authorize a GPU campaign, or establish novelty or a performance advantage.
The user requested a concrete design and recommendation on 2026-11-02.

## Required computation and useful guarantee

Consider ordered mechanisms `m`, cellular contexts `c`, participant occurrences
`P_m`, restricted conditioning inputs `C_m`, and output destinations `T_m`:

```text
r[m,c] = F[law_m, parameter_binding_m](x[c,P_m], z[c,C_m])
y[c,t] = reduce contributions r[m,c] for which t is an output destination
```

The first probe uses additive output reduction and explicit functional input and
output generations. A transition that mutates values in place has a different
ordering contract and cannot silently use this evaluator. The law can include
a learned gate. Separate numerical eligibility is optional and has an explicit
forward/backward contract.

The guarantee to exploit is *repeated access to overlapping ordered participant
and conditioning sets within one evaluation*, plus reuse of the candidate
structure across evaluations and cells. Mutable activations are not reusable
across training steps merely because topology is unchanged. Shared participants
permit shared loading; shared arithmetic requires an additional exact operator
factorization and compatible parameter bindings.

The inherited convention being challenged is one generic feature tensor plus
a flat list of edges that repeatedly gathers its operands. A tensor-core tile,
CSR row, or biological module is not a required physical unit.

The protected [cellular geometry](../../../cellular_geometry.md),
[scope](../../../scope.md), and
[virtual-model architecture](../../../docs/architecture.qmd#virtual-models-and-structural-specialization)
remain governing authority. Missing observations do not establish exclusion.
The layout preserves the candidate mechanisms; choosing candidates is a
separate model decision.

## Recommended representation

Partition mechanisms into small packets according to compatible operator law,
arity/shape, precision, and overlap of actual reads and writes. One packet might
contain 16-32 mechanisms, with bounded distinct input and output occurrences.
These are initial search choices, not fixed architectural dimensions. Mechanisms
may be reordered; ordered roles and repeated occurrences retain their meaning.

A packet's physical records are arrays in bulk storage, not heap objects:

| Array or descriptor | Meaning and lifetime |
| --- | --- |
| `mechanism_id[slot]` | Stable logical IDs in packet execution order. |
| `input_occurrence[local]` | Dictionary of distinct numerical input occurrences/views read by the packet. |
| `role_ref[role,slot]` | Small local dictionary indices; preserves ordered roles and repetitions. |
| `context_occurrence[local]`, `context_ref[role,slot]` | Separately factored conditioning reads. |
| `parameter_binding[slot]` | Reference to the existing shared parameter owner, including the relevant slice/view. |
| `output_ref[role,slot]` | Local output destinations and exact contribution mapping. |
| `output_postings`, `input_postings` | Grouped forward contribution and reverse derivative routes. |
| packet/operator descriptors | Extents, offsets, law/shape/precision bin, structural epoch and identity/layout maps. |

Initially use 8-bit local refs only for packets whose dictionary and sentinel
encoding fit; use 16-bit or split packets otherwise. Global occurrence references
and stable mechanism IDs keep their required native widths. Ref compression does
not narrow biological IDs or merge two events involving the same gene.

The differentiating physical choice is to factor *argument columns*, rather than
pad a two-dimensional adjacency rectangle. A role shared by all mechanisms is
stored once as a broadcast slot. Mostly shared roles can use a common slot plus
an exception list when decoding is worthwhile. The first executable probe can
use the plain role-major local-ref array before comparing the extra compression.
There is no proposal for a runtime interpreter executing arbitrary per-mechanism
function pointers: each compatible bin uses a concrete numerical evaluator.

### Worked encoding

Four mechanisms of the same three-input law have different parameters:

```text
m0: F(theta0; a,b,c | z0) -> u
m1: F(theta1; a,b,d | z0) -> v
m2: F(theta2; a,b,e | z1) -> u
m3: F(theta3; a,a,c | z0) -> w

input dictionary:   [a,b,c,d,e]
role 0:             [0,0,0,0]       -> broadcast a
role 1:             [1,1,1,0]       -> b, with one exception
role 2:             [2,3,4,2]
context dictionary: [z0,z1]
context refs:       [0,0,1,0]
output dictionary:  [u,v,w]
output refs:        [0,1,0,2]
parameter bindings: [theta0,theta1,theta2,theta3]
```

This encodes five distinct participant values for twelve participant reads, and
two conditioning values for four conditioning reads. Both reads of `a` in `m3`
remain present in the law and its derivative. The gate for `m0` cannot be reused
for `m1` merely because both read `z0`: the function and parameter binding must
also match. Equal numerical values are not an identity-equivalence rule.

### Numerical storage and producer/consumer boundary

For compatible numeric inputs, retain a model-specific physical state order with
tiles such as `values[occurrence_tile, channel_tile, cell_lane]`, with cell lanes
contiguous. A packet dictionary names occurrences in that order. A CTA stages
the required occurrence/channel/cell tile once and lets its mechanisms reuse it.
Different packets may stage the same occurrence independently.

The producer may directly write this physical order if its mathematical output
supports it. Otherwise count an explicit boundary conversion. Retain the order
through compatible downstream operators; materialize canonical order at a real
external boundary or when a better conventional operator requires it. This is
an optional numerical representation, not a universal Cell or tensor contract.

Packet structure and operator descriptors survive compatible value changes.
Staged activations, gates, projected values, and numerical summaries have actual
invocation/parameter-generation lifetimes. Parameters stay with the existing
owner; any packed parameter copy needs an explicit refresh and adjoint mapping.
Concurrent evaluations and autograd tapes retain their own live values or valid
recomputation inputs.

## Execution construction

1. Build bins for compatible law, arity/shape, precision, and reduction contract.
   Within each bin group mechanisms by weighted participant/context overlap and
   output locality, subject to dictionary and live-state budgets. Candidate
   clustering cost is part of preparation. Distinct functions with the same
   participant set do not automatically share arithmetic.
2. Launch a grid over packet, cell tile and channel/output tile. Cooperatively
   load the packet's unique input/context dictionary. Local role refs address
   shared-memory/register tiles, rather than edge-sized global input copies.
3. Evaluate mechanisms in parallel. For a simple elementwise joint law, one warp
   can cover eight cells by four channels, loop over channel tiles, and iterate
   over mechanisms assigned to that warp. This is a candidate mapping, not a
   universal mapping for MLPs, attention, or arbitrary learned operations.
4. Reduce repeated output contributions inside the packet. Write unique packet
   partials, then use prepared output postings for a second reduction when
   several packets contribute to one output. Alternatively measure direct
   atomics under an explicitly permitted numerical/reproducibility contract.
   Counts, shared scratch, barriers and partial buffers must fit the real budget.
5. Backward uses the same logical roles and a prepared reverse route. It adds
   gradients from every role occurrence and packet into the correct input and
   parameter owner. A packet-local dictionary gradient is accumulated before
   writing unique partials; cross-packet reduction is explicit. Shared storage
   is not shared gradient ownership and does not imply gradient deduplication.

One hypothetical V100 tile with 64 unique FP16 inputs, 16 channels and eight
cells requires 16 KiB for input staging. A distinct FP32 output tile for 32
outputs requires another 16 KiB; 32 scalar FP32 gates across eight cells add
1 KiB. This is about 33 KiB before alignment, refs, reduction scratch and any
extra context staging. Count context inside the input budget or add it explicitly.
Reusing output/gate storage can shorten live ranges. A small arithmetic operator
could plausibly fit, but arbitrary MLP activations can break this budget. Actual
compiled register/shared-memory use and occupancy remain unmeasured. No special
Tensor Core instruction, persistent kernel, or SM-resident data lifetime is
assumed. NVIDIA documents the relevant Volta resource and participation rules in
its [tuning guide](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html).

### Conditioning and eligibility

The context dictionary makes conditioning reads and invalidation dependencies
explicit. A reverse context-to-packet index can identify affected conditions
when the model provides valid change information. This is not automatically an
incremental-training algorithm: every declared changed dependency and the
autograd graph lifetime must be honored, and updates may make all packets dirty.

Continuous gates are computed numerically and differentiated. Exact eligibility
can produce masks over mechanism slots and cell lanes; compact whole work groups
only when both primal and derivative obligations permit omission. A zero product
or a small sigmoid is not a certificate of zero derivative. Masks themselves,
condition evaluation, and compaction have costs. Membership or role changes
require a new structural snapshot or explicitly reconciled local replacement.

### Attention specialization

Attention is a specialization of this representation, not its definition.
Queries, keys and values are separate numerical views in dictionary identity.
A neighborhood split among packets must retain a mergeable summary for each
target/cell/head: maximum logit `m`, shifted exponential sum `l`, and weighted
numerator `u`. Merge summaries with rescaling to the common maximum, then return
`u/l`. Empty neighborhoods use an explicit empty state and return zero.

Normalizing each packet independently changes attention and is invalid. Backward
must differentiate the complete merged neighborhood or recompute against saved
row normalization. Dropout, if later admitted, needs stable logical edge-based
RNG mapping and its own numerical contract. The first probe has zero dropout.

## Why it might help, and what it cannot claim

Let `I` be ordered participant incidences, `U_p` the unique input occurrences in
packet `p`, `L = sum_p U_p`, `B` evaluated cells, `D` channels and `b` bytes/value.
Ideal staged input payload is roughly `L*B*D*b`, compared with an explicit
role-expanded input payload of `I*B*D*b`. Local refs and repeated role columns
also reduce structural traffic. This is an opportunity bound, not measured HBM
traffic: a good existing gather kernel may already reuse inputs in cache.

As a deliberately favorable illustration, 32 four-input mechanisms that use
16 unique occurrences have 128 incidences. A tile with eight cells and
16 FP16 channels holds 4 KiB of distinct input values versus 32 KiB of
role-expanded values. With 32-bit global refs and 8-bit local refs, reference
storage is 64 + 128 = 192 bytes versus 512 bytes of global refs. Packet offsets,
stable IDs, context, output, transpose and parameter metadata are additional.
These ratios compare representations; neither is a claimed speedup.

The proposed layout removes no arithmetic for arbitrary incompatible laws.
It does not cache mutable values across optimization steps. It does not remove
cross-packet input duplication, output contention, backward reduction, boundary
conversion, or parameter packing. Low overlap (`L` near `I`), large live state,
one-off evaluations, highly heterogeneous laws, and rapidly changing membership
are plausible losing regimes. Dense or ordinary sparse/library routes may win.

The complete decision is whether:

```text
preparation + boundary conversion + N * (forward + backward + reductions
                                      + value/parameter maintenance)
```

is better than the strongest semantically equivalent conventional route over
the caller's actual reuse interval. Repeated reuse cannot rescue a slower
recurring path merely by increasing `N`.

## Serious alternative: producer-written ordered-role slabs

Instead of a packet dictionary, materialize each mechanism's role values in
`operand[packet,role,channel,cell_lane,mechanism_slot]` or an appropriate tiled
permutation. A participant-owned fanout map lets upstream producers write the
consumer slots directly. The consumer then has fixed offsets and regular
loads, and a backward fan-in map sums gradients to the actual source owner.

This is a more radical producer/consumer storage change, but duplication is
expensive: publishing and reading `I` role copies is approximately `2I` value
traffic before adjoints, while dictionary staging ideally reads `L <= I`.
Retaining a separate canonical output adds more. Compare this candidate only
where fused producer writes, repeated consumers, sparse updates, or materially
better arithmetic utilization could repay publishing, residency and adjoint
costs. Global broadcast-like fanout and fully changing values are likely poor
fits. No benefit is claimed from eliminating indirection alone.

The lead probe is dictionary-backed packets with an explicit producer-compatible
value order. Producer-written role slabs are the serious alternative, not a
mandatory second implementation or default replication policy.

## First discriminating experiment

The first operator is an explicitly declared gated joint product or small joint
neural function with additive outputs, across many cells sharing candidate
structure. Include ordered repeated roles, overlaps, independent parameters and
explicit conditioning. The initial numerical candidate uses FP16 input and
parameter storage, FP32 multiply/gate/reduction arithmetic and gradient
accumulation, and FP32 output, with any FP16 output cast tested separately.
Respect explicit caller precision. Compare to the same law in its ordinary
upstream/native precision; a higher precision oracle supplements that comparison.

Hold the equation, candidate mechanisms, input/parameter values, objective and
precision fixed. Compare a strong Torch/ATen gather/scatter route, a parallel
CSR/role-list implementation, and an applicable blocked/library route. The
known serialized staged-attention kernel is not the sole performance baseline.
For attention use a matching supported sparse/block-sparse attention baseline
such as FlexAttention where available and contract-compatible.

Sweep participant reuse, context reuse, output fan-in, arity, feature width,
cell count and structural churn. Include random low-overlap negatives,
shared-core mechanisms, overlapping motifs and a supplied real candidate graph
when its data and scientific scope are selected. Model similarity does not prove
biological validity. Separate topology reuse from freshly changing values and
parameter updates on every training iteration.

Reject or revise the candidate if it fails exact identity/role reconstruction,
forward/VJP agreement within the declared numerical contract, zero-input and
gate derivatives, duplicate-output/repeated-role accumulation, or safe tape
lifetimes. Test attention segmentation/merge invariance separately. Measure
preparation, conversion, persistent/scratch memory, complete forward+backward
cost and maintenance; pair measurements under existing admission/interference
controls. If high-overlap shapes lose after those costs, the reuse mechanism
has failed its first performance discriminator. If only low-overlap shapes lose,
retain a bounded applicability rule rather than universal promotion.

This document launches no benchmark. An implementation/probe task would use the
repository's current authority, Autobench, KDA evaluation as relevant and Device
Scheduler admission. A layout-only feasibility probe does not establish a
whole-model biological or training advantage.

## Existing seams and prior art

Current [ExplicitPattern](../../../include/Cellerator/geometry/dependency/explicit_pattern.hh)
already preserves ordered participant occurrences and stable mechanism IDs.
Current [NN attention](../../../src/nn/attention.cc) consumes prepared CSR and
does not implement the packet layout above. Reuse compatible identity, storage,
parameter and derivative owners; a new representation is not a new model
superclass, universal tensor facade or another general planner.

Tiles, hyperedge grouping and composable sparse formats have prior art.
[HyperGef](https://proceedings.mlsys.org/paper_files/paper/2023/file/895783891e093ded4f0c48c250fde92a-Paper-mlsys2023.pdf)
develops fused hypergraph work and shared-memory-aware grouping;
[SparseTIR](https://arxiv.org/abs/2207.04606) develops composable sparse formats
and transformations;
[FlexAttention](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html)
provides block-mask-driven attention; and
[SAM](https://compilers.stanford.edu/publications/asplos23-sam/)
separates sparse formats from scheduled dataflow.

The proposed distinction worth investigating is the combined physical treatment
of ordered/repeated roles, factored restricted context, shared parameters,
cell-lane values, and adjoint routes for general conditional joint mechanisms.
The targeted source scan is not a novelty review, and no claim of globally new
layout design is made.

## Evidence and design guidance

Evidence class: source review, proposed encoding, arithmetic estimates and
independent design consultation. No numerical implementation, compilation,
GPU run, profiler capture or performance measurement belongs to this proposal.
The root retains the recommendation and acceptance; agent suggestions are
advisory. No canonical design document or existing numerical behavior changed.

Source snapshot: repository HEAD `b6f1a812c369c9d60fa45eb3c3df1c12f2d5651e`,
with concurrent uncommitted work. Relevant observed SHA-256 identities:

| Input | SHA-256 |
| --- | --- |
| `cellular_geometry.md` | `d37f504e5c1ab629334f4127dcaa60d008afe94e560d41aa936d8ba161124254` |
| `include/Cellerator/geometry/dependency/explicit_pattern.hh` | `aaf47be9e11e9a9646e899adde31ccf864ce503d2ede87757b0a2fb76496535b` |
| `src/nn/attention.cc` | `7a1decf97b11dbee30ed23382ad8346a03a787ce817a3a29e6d1a66f6b05ff11` |

Consumed design guidance: CUDA skill and machine-aligned-design guide, KDA
entrypoint, KernelBender and its identity/ownership/lifetime/summary/parallelism
bridges, experiment guidance, and the Volta resource guidance. KDA observed
revision: `277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc`.

| Guidance entry | SHA-256 |
| --- | --- |
| `/home/tumlinson/.agents/skills/cuda/SKILL.md` | `3992c5b0e85409cce8002fcfe2c6caed63df6451daad9bae28ccd64a381d717a` |
| `/home/tumlinson/.agents/skills/cuda/references/common/machine-aligned-design.md` | `78b489b241c5be8907f36f8f26e70edd39c9d4bfb26e337d0fe7eb0ce693e779` |
| `/home/tumlinson/kda/docs/agent-entry.md` | `e7f85c69274b6524c5fc4e13f3cffec8c7d42469f23a4e1e1c0555c47451bbc6` |
| `/home/tumlinson/kda/skills/KernelBender/skills/kernelbender/SKILL.md` | `e00a21a31493af473a6e341ae745b042221eff9543483d16c4f7c83fde776347` |
