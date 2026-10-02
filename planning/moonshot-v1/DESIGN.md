# Design: make one cell a useful matrix program

## 1. The actual target

Make an evolving cell cheap to evaluate and differentiate, both as a plain learned model and as a model whose internal computation can gradually be decomposed. The target is not expression compression, not a compulsory causal graph, and not an obligation to give every actor sixteen hidden numbers.

The reference idea from the conversation is retained: actors may carry local polysemantic hidden state. An external learner can factor supported structure into another actor or meta-actor. The original actors retain residual state and can reuse liberated capacity. The factor need not be a molecule. No latent coordinate needs a global scientific name. The model may be reorganized while learning rather than interpreted only afterward.

The exact data structure, factorization, slot budget and extraction procedure remain open. The mechanism is a reason to support cheap reorganization, not a requirement to build a dynamic object graph or a global hidden dictionary.

## 2. Three moves that must not be confused

**Equivalent lowering:** compute the supplied function using a different schedule, factor representation, memory order or instruction. Preserve its inputs, weights, contribution ownership and declared numerical policy.

**New model family:** offer a function whose factorization makes useful mathematics cheaper. A matrix patch is an example. It may be a very good learned model, but it is not an equivalent replacement for every arbitrary dense or sparse map.

**Approximate execution:** deliberately skip, quantize, interpolate or approximate work. Keep the approximation visible and its derivative convention explicit. The first prototype can be empirical; it must not advertise an unproved error bound.

These labels permit bold implementation without confusing a mathematical change with a faster kernel. They are not a demand for biological qualification before coding.

## 3. The central correction: a scalar vector can still define matrix-native dynamics

Take 256 existing state coordinates from one cell. Rearrange them bijectively into `X[16,16]`. There is still exactly the same amount of state. Now choose, for example,

\[
Y=LXR.
\]

This is a genuine pair of matrix products. Each arithmetic axis has useful work. The state matrix is not sixteen cells and its columns need not be globally named hidden regulators.

The important move is choosing the **operator**, not merely reshaping the input. With column-major vectorization,

\[
\operatorname{vec}(Y)=(R^T\otimes L)\operatorname{vec}(X).
\]

Thus the model represents a structured 256-to-256 map. With square 16-wide factors it uses 512 factor coefficients and 8192 multiply-accumulate terms, versus 65536 coefficients/terms for an unrestricted dense linear map. These are algebraic counts, not measured speedups or equal-expressivity claims. Several terms, nonlinearities, sparse residuals and changing patch permutations can broaden the family.

Structured matrix models such as tensorized networks and Monarch are precedents for parameterizing useful transforms rather than insisting on an expanded dense matrix [R5, R6]. This proposal is not a claim to reproduce Monarch.

**Recommended primitive family:** `LXR`, `tanh(LX)R`, `LX+XR`, `XMX`, and sums/compositions of these with explicit sparse exchanges. Implement the algebra and adjoints once; different models choose which members make scientific sense.

## 4. A stronger trajectory move: compile a flow, not only a derivative

For the autonomous local law

\[
\dot X=LX+XR,
\]

with fixed `L,R`, the flow is

\[
X(t+\Delta)=e^{\Delta L}X(t)e^{\Delta R}.
\]

Differentiate this expression to verify the law. It is not the solution of `dot X=LXR`. It gives an unusually useful learned family: many numerical substeps can be replaced by a directly evaluated linear flow. If parameters and step size remain fixed, the two exponentials can be prepared and reused. Parameter gradients must still differentiate their dependence on `L,R`; a detached cache is not a valid training shortcut.

Add local nonlinear reactions or exchanges through a declared splitting scheme, or learn a residual step map. That extension is a numerical/model approximation, no longer the exact full-system flow. The first prototype should simply expose the exact linear-flow block. A smaller exact piece can be valuable without claiming the whole cell is linear.

For optional dissipative latent models, explore `L=K-K^T-UU^T-gamma I` and a similar `R`; positive gamma yields a contraction argument for the linear part in the Frobenius norm. This is a restriction on latent dynamics, not a universal property of biology. Other models may need oscillatory or amplifying regions.

## 5. A second representation: local coordinates with communication ports

Keep each actor's `h_i` local and arbitrary. Define

\[
m_i=E_i h_i,\quad z_i=\sum_j A_{ij}m_j,\quad
\dot h_i=\phi_i(h_i)+D_i z_i.
\]

The messages have a common computational interface. The private hidden coordinates do not have a common meaning. Under `h'_i=Q_i h_i`, use `E'_i=E_i Q_i^{-1}` and `D'_i=Q_i D_i`; the transport law changes coordinates correctly. The local nonlinear law must also be conjugated, and an arbitrary fixed neural architecture need not be closed under that conjugation.

This directly repairs the mistaken earlier suggestion that all hidden dimensions should be canonicalized. A port is closer to a function argument than a globally identified biological regulator.

The incoming-message matrix can have genuine width within one cell. Actor-local encoders with different weights remain separate operations: batching their input rows does not magically make all their weights identical. Either lower them as micro-operations, or explicitly choose a shared dictionary of local maps as a model restriction. Program sharing, parameter sharing and value sharing are distinct permissions.

## 6. Volta offers a smaller unit than the familiar WMMA tile

The documented FP16-input, FP32-accumulator `mma.m8n8k4` layout computes four independent `8x4 * 4x8` products in a warp [R2]. That makes four different local contractions an interesting packet. They can have different weights. Full-warp execution is still required: these are not four independently launchable eight-thread tensor cores.

A prewritten inline-PTX seed implements the documented operand and result mappings. Its CPU check verifies that the output mapping covers each result once. Only execution on SM70 can verify the actual instruction binding; that test is supplied but not run here.

This is useful for local maps that genuinely have several output quantities, small response panels, quadratic factors and partial contractions. It does not remove the missing-column issue of an arbitrary isolated matrix-vector product. DASP nevertheless shows that a categorical ban on Tensor-Core SpMV is too strong [R8]. Treat such embeddings as further candidates, with explicit redundant-work and load costs.

## 7. Compile compute regions, not a second object runtime

A compute region has a declared input snapshot, ordered computations, owned output contributions and derivative requirements. It might be one patch, several related actors, a shared process, or a small stiff feedback island. A region can span many tiles and contain dense and sparse paths.

Three representations are enough for the first implementation:

**Logical description:** actor-local coordinate IDs, supplied relations/mechanisms, readout roles and provenance.

**Prepared projection:** index maps, tile payload order, sparse remainder, output ownership, kernel choices and support footprints.

**Bound runtime state:** current values, parameters, activity, stream, workspace and saved-primal generations.

These should adapt to the existing Cellerator memory, operation, program and generation interfaces. The payload's `state_packet.hh` is a physical representation seed, not an alternative allocator or replacement public ABI.

A frozen sparse state can use canonical flat coordinates with explicit actor/slot maps. A more tensorized model can use patch-major values. Several physical projections are allowed when they earn their maintenance cost. Physical copies always carry the canonical state identity and generation; their adjoints reduce back to that owner.

## 8. Packing follows the computation

For placement, compare input, output and activation footprints in explicitly tagged universes. Do not mix measurement-detection bits with dependency bits. Similarity nominates a pair of regions; it does not establish that their latent states are equal.

A useful packing objective includes tile padding, gathers, shared inputs, output reductions, parameter traffic, derivative traffic, active fraction and representation-change cost. The existing MinHash/LSH and coarsen/refine/oracle pattern is reusable conceptually. The delivered CPU prototype uses a small deterministic heuristic rather than pretending to reproduce the native optimizer.

Pack at least two different things separately: state access and operation execution. Two actors can share consumers but require different nonlinear evaluators. Conversely, unrelated actors may have exactly the same opcode and be excellent lane neighbors without being scientifically related.

The packed order is not the scientific hierarchy. Support similarity, biological interpretation and instruction shape have different jobs.

## 9. Learning-time refactoring is a publication event

Cellerator should accept an explicit rewrite from the learning layer: input model epoch, extraction or coordinate map, new declarations, state initialization, parameter/readout migration, residual-capacity policy and evidence label. The label may say `engineering_hypothesis`, not `identified_mechanism`.

Start with supplied maps. Do not block the prototype on discovering an unknown regulator.

For an invertible change `q=T s`, preserve a vector field by `G(q)=T F(T^{-1}q)`. Readouts transform too. For a reduction `s=E q`, require the retained manifold to be invariant, not merely equality of two coordinates at one observed point. The linear reference checks `PE=I` and `AE=E(PAE)`.

Wait for old readers/tapes to finish before publishing a structural epoch. A recycled slot gets a new generation, so stale physical addresses cannot accidentally refer to new meanings. Parameter masters, optimizer states, buffers and graph references must move coherently. Diagonal Adam moments are not generally invariant under a dense change of basis; use an explicit reset policy rather than inventing a universal moment transform.

To make freed capacity learn immediately, a safe local reference is `r(x)=V tanh(Ux)`: seed `U` nonzero and set `V=0`. The initial output is unchanged, but the outgoing gradient is generally live. Setting both maps to zero can make the branch unable to learn. A newly introduced recurrent state also needs an initialization and evolution law; this feed-forward witness is not a universal stateful morphism.

Network morphism and dynamic sparse training offer relevant precedents [R9–R11], not evidence that the resulting actor is a unique biological mechanism.

## 10. Resident evolution and change-driven work

Keep state, parameters and intermediate results resident across operations. Begin with a prepared static sweep. Dynamic worklists are optional where activity or changes are sufficiently sparse to pay for bookkeeping.

For a fixed linear map, maintain `y=W x_sent`. Applying deltas gives `y += W[:,changed] delta`. Threshold against the **last transmitted** value; comparing only consecutive timesteps can lose a long accumulation of individually small changes. Parameter changes invalidate this cache. Floating-point update drift may require periodic refresh. The delivered bound is instantaneous linear-output error, not a whole-trajectory bound [R12 is a related recurrent-computation precedent].

Distinguish forward work from response work. `a*b` at `a=0` can contribute zero while its derivative with respect to `a` is `b`. The existing NF1 contract already preserves this distinction. Avoid writing a fast gate that deletes GlassHelix's requested sensitivity.

## 11. Prototype order

Build matrix patches, micro-MMA and local-port transport first. They exercise different computational hypotheses while sharing existing state/parameter ownership. Add explicit promotion and live residual recycling next, using supplied maps. Then build delta execution and one of matrix flow, feedback islands or texture-rate lookup as a distinct experiment.

A successful sprint produces callable native prototypes and two consumers: one plain prediction model and one externally refactorable hidden-state model. It does not need a large biological cohort, an automatic scientific extraction algorithm, or an exhaustive timing report.

Prefer six to eight substantive engineering outcomes over hundreds of procedural TODOs. Use small algebra/gradient/shape checks, one useful smoke model, and a short receipt saying what is unsupported. Do not turn curiosity into a qualification bureaucracy.

## 12. What would count as a genuinely new result

Not merely running an old graph through a different sparse container. A useful result could be a scalar-state operator that gets genuine Tensor-Core reuse; an exact local flow that eliminates a class of timesteps; a per-actor model whose hidden bases stay private; a structural rewrite that returns trainable capacity without losing the old function; or a mixed instruction schedule where communication costs less because topology is encoded in lane maps.

The unifying aim is **an evolving cell represented as a compact executable mathematical program, with both its algebra and its physical layout open to improvement**.
