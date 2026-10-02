# Mathematical extension cards: precise functions, useful compositions

These are original integration proposals and derived identities, not claims of a discovered biological mechanism or measured GPU speed. They extend and connect the existing M01–M13 and E01–E48 programmes; they do not replace them. All general mathematics belongs in Cellerator. Baseplane and GlassHelix supply distinct consumers and scientific/sequence meaning. Implement a new primitive through the integrated operator interface rather than creating a permanent `new_experiment_17` path.

Labels: **E** equivalent for the declared mathematical function; **M** a model-family choice; **A** approximation with an explicit policy. Floating-point association can differ even when a real-arithmetic identity is exact. For every implemented member publish shapes, domain, precision, forward/derivative capabilities, support footprint and invalidation conditions.

## N01 — An effect that also remembers what happened

**E/M.** A regional computation acts on state h and accumulates an observable q:

    h' = A h + b
    q' = q + C h + d

A left region followed by a right region composes into:

    A = A_R A_L
    b = A_R b_L + b_R
    C = C_L + C_R A_L
    d = d_L + C_R b_L + d_R

This carrier is closed for these affine laws. It remembers integrated linear rewards/readouts or counts represented continuously while preserving order. It unifies a state transform and the effect it has on an accumulated measurement without replaying every intermediate step. The additive q convention is explicit: an arbitrary q-dependent dynamics is a different carrier.

**Derivative:** dh' = dA h + A dh + db; dq' = dq + dC h + C dh + dd. With output cotangents u,v: bar_h=Aᵀu+Cᵀv; bar_q=v; bar_A=u hᵀ; bar_C=v hᵀ; bar_b=u; bar_d=v. Differentiate composition by the product rule. A prepared prefix/suffix scan can supply requested boundaries, but ordinary floating-point composition is not bitwise associative.

**Machine mapping:** small matrices use actual panel width; monomial/diagonal subfamilies use gathers/shuffles and scalar operations. Choose a carrier per useful region, not per base. Store readout coefficients only for genuinely requested readouts. A fixed observation family can enable cached composition; changed parameters or readout maps invalidate it.

**Consumer:** BP answers an incoming-state question plus a regional accumulated response; GH evaluates a declared piecewise-affine latent model with cumulative observables. This is not a general exact nonlinear trajectory solver.

**Failure test:** compare direct sequential application against composed application, including noncommuting A matrices, nonzero initial q and empty identity. Compare memory/preparation cost against replay as region length and repeated query count vary. Reference: `effects.py:AffineWitness`.

## N02 — A discrete routing state carrying a continuous transformation

**E/M.** Define a finite-control affine effect:

    F(s,h) = (t[s], A[s] h + b[s])

For left then right:

    t[s] = t_R[t_L[s]]
    A[s] = A_R[t_L[s]] A_L[s]
    b[s] = A_R[t_L[s]] b_L[s] + b_R[t_L[s]]

The result remains in the family. Combine a finite grammar/context state with a continuous latent response rather than choosing between a Boolean automaton and a dense neural state. This is a conditional family indexed by the incoming control state, not an inference that every genome admits 32 useful states.

**Machine mapping:** for 32 control states, a warp can hold transition targets and use shuffles to select right-hand parameters. Scalar/monomial continuous subfamilies fit cheaply; dense per-state blocks can require several warps and substantially more storage. The supplied CUDA seed implements only one continuous scalar per state as a precise, small starting point. It uses no tensor core merely for decoration.

**Derivatives:** continuous weights are differentiable along the selected route. Fixed discrete transitions do not have an ordinary gradient with respect to their integer values. A relaxed teacher, straight-through rule or explicit route search is a separately labelled training policy. Version the vocabulary and any hardened transition table.

**Consumer:** a sequence region whose effect depends on bounded entry context; a CE latent mode-switching model chosen by GH. Extend N01 with per-state C,d only when a real query needs it.

**Failure test:** all incoming states, three-region associativity, non-bijective transitions, state-vocabulary mismatch, invalid inputs and no hidden state truncation. Reference: `effects.py:HybridAffine`, `seed/cuda/hybrid_scalar_sm70.cu`.

## N03 — Response jets as first-class cached mathematical objects

**E for derivatives at the recorded point; A for finite perturbation prediction.** A first-order response object stores `(x0, y0, J, parameter_generation, definition_id)`. For f then g, with g expanded at y0=f(x0):

    y0_composed = g(y0)
    J_composed = J_g J_f

The derivative at x0 is exact under the declared differentiability assumptions. Predicting a finite change by y0+J*delta is a local approximation. A second-order directional response adds `J_g H_f[u,v] + H_g[J_f u, J_f v]`; it must not be replaced by applying g independently to tangent columns.

**Machine mapping:** only requested tangent directions form panels. A skinny Jacobian/response representation may reduce both work and storage; a full dense Jacobian is not mandatory. Group compatible derivative operations, not arbitrary physical times. Save or recompute the same primal intermediates used by the chosen derivative convention.

**Consumer:** BP revisits a regional response outside its explicit expansion/trust context; GH asks for local sensitivities under specific hypotheses. A learned confidence gate is not a guaranteed Taylor error bound. Existing CE local differential facilities are the integration point.

**Failure test:** wrong expansion point, changed parameters, nonlinear finite changes and arbitrary tangent scales. Reference polynomial JVP/VJP and adjoint checks in `operators.py`/tests; native jet scheduling is implementation work.

## N04 — Exact change propagation for matrix-polynomial dynamics

**E for fixed coefficients in real arithmetic.** For F(X)=LX+XR+XMX and a change D:

    F(X+D)-F(X) = LD + DR + DMX + XMD + DMD

The last term is essential. Omitting it is a linear approximation, not exact incremental execution. For D=uvᵀ:

    DMX = u (vᵀ M X)
    XMD = (X M u) vᵀ
    DMD = u (vᵀ M u) vᵀ

A low-rank changed state can therefore induce a compact exact update without recomputing every dense term. For higher-rank D=UVᵀ the same factoring applies, but rank and intermediate growth must be charged. No cheapness follows when changes are dense or frequent.

**Derivatives:** JVP in state direction P is LP+PR+PMX+XMP. The VJP and parameter adjoints are implemented in the reference. Incremental forward caches do not automatically provide a correct backward tape; either retain the declared primal snapshot or recompute it under the right generation. Cache against last transmitted state. Refresh on changed L,R,M and periodically for floating recurrence drift.

**Consumer:** resident CE nonlinear updates; BP contextual response changes; GH changing latent state. This extends the prior linear delta ledger rather than creating a separate event simulator.

**Failure test:** several subthreshold changes, a changed coefficient generation, arbitrary rank perturbations, X=0 with live derivative, and dense-change crossover. Reference: `operators.py:quadratic_delta`, `QuadraticLedger`.

## N05 — A region exposed as boundary ports with interior recovery

**E for a supplied linear system; M/A for a learned or linearized surrogate.** Partition a linear system into boundary b and interior i:

    A_bb x_b + A_bi x_i = r_b
    A_ib x_b + A_ii x_i = r_i

When the required solves are well posed, expose:

    S = A_bb - A_bi solve(A_ii, A_ib)
    g = r_b - A_bi solve(A_ii, r_i)
    x_b = solve(S,g)
    x_i = solve(A_ii, r_i - A_ib x_b)

The interior is not erased; it is recoverable under the same coefficients/RHS. A boundary response can be cheaper to compose repeatedly than a full internal model. Factorizations are keyed to parameter/structure generations. Test conditioning and residuals, not just equality of small outputs.

**Derivative:** for A x=r, dx=solve(A,dr-dA*x). A VJP uses a transposed solve. A deliberately truncated iterative solve has a different differentiated computation unless implicit differentiation is explicitly the chosen convention. Reuse CE's solve/feedback machinery and factor storage.

**Machine mapping:** narrow boundary panels and many small interiors may exploit batched factorizations and true tensor-core contractions where precision permits. Do not form an unstable matrix inverse because it looks like GEMM. A low-rank island and a Schur boundary are complementary factorizations, not synonyms.

**Consumer:** a BP region with learned/supplied ports, or a GH local feedback hypothesis. The port choice and linearization are scientific/model assumptions.

**Failure test:** reordered boundary indices, nearly singular interiors, changed RHS versus changed coefficients, recovery of interior and JVP against a full solve. Reference: `operators.py:schur_solve`, `solve_jvp`.

## N06 — Residual-backed multilevel mathematical operators

**M, or E only for an explicitly supplied decomposition.** Keep a cheap coarse map and an independently useful fine residual:

    y = D x + P K R x

R restricts, K acts on coarse coordinates, P prolongs, D supplies local/fine structure. This permits meaningful coarse matrix work without forcing a dense global map or claiming every operator is low-rank. Several patch/coarse terms and a sparse D broaden the family. In a fixed declared linear family the application and derivatives are exact; replacing an arbitrary operator by it is a model restriction or approximation.

Combine it with reversible lifting so coarse state can travel while detail remains recoverable. Learned restriction/prolongation need not be inverse or orthogonal unless the model declares those properties. Refinement can request residuals selectively; retaining them all saves movement/work, not automatically storage. Rank or patch repartition changes use the same epoch/migration machinery as other structural changes.

**Machine mapping:** local/sparse D, narrow R/P panels, small K tiles. Compare against the actual direct map including gather/scatter and residual bytes. The response uses the transposed factors in reverse order; shared parameter gradients keep their logical ownership.

**Failure test:** weak/strong coarse signal, large residual, ill-conditioned learned bases, requested local observables and extension to ragged actors. Reference: `operators.py:multilevel_apply`; native integration is an open family, not a required global solver framework.

## N07 — Compile the derivative demand, not a compulsory full Jacobian

**E for requested derivatives.** Describe primal computation and separately requested outputs/directions. Construct the backward-needed and forward-response-needed subgraphs with structural derivative rules. Reuse intermediates where they are valid; pack primal and direction panels only for operations whose algebra permits it. A product at a zero argument can have zero primal contribution and nonzero derivative; ordinary activity masks cannot erase response edges.

Use opcode cohorts and generation-bound saved-primal maps to make this a prepared program. A small set of explicit operator JVP/VJP rules is enough initially. Differentiation of arbitrary user CUDA is not promised. Custom operators register supported response rules and can reject unsupported requests without breaking the rest of the library.

**Consumer:** GH perturbation/ambiguity analysis and learning; BP query-conditioned responses. Derivative work should help fill genuine matrix axes, never invented tangent columns that are not used. Compare full response versus demanded response and include preparation/recording overhead.

## N08 — Jointly optimize what is adjacent and what executes together

**E when only schedules/layouts change.** Use two independent variables: a state-access placement and an operation-cohort placement. Alternate cheap local improvements under a measured or explicitly hypothetical total-cost objective. The best co-location for shared source loads may differ from that for common opcode, shared parameters or output reduction. Do not declare a biological module merely because the machine favors the group.

An admissible move preserves ordered arguments, contribution ownership and capabilities. Try the existing exact occupancy oracle for compatible relation layouts; use a different estimator for process/port packets that do not fit Cellpack's format. Retain the best accepted plan and a no-pack fallback. Plan publication and migration costs enter the horizon test in `03_PACKING.md`.

**Consumer:** the same CE program under user-supplied, Cellpack and operation-oriented plans; BP queries whose active detail changes; GH training with different forward/response mixes. The supplied packer reference demonstrates the independence of state order and work order, not a winning optimizer.

## N09 — Share counterfactual computation without mixing hypotheses

**E for identical subcomputations with complete guards; A for threshold-based approximate reuse.** Let several complete worlds reference immutable structure and carry sparse state/parameter/sequence deltas. Share only a subexpression whose definition, arguments, versions and relevant context are identical. Divergent worlds split that subgraph; reconvergence requires an explicit equality/query-equivalence condition.

The object being shared is computation, not a coordinate-wise mixture of scientific alternatives. A repeated sequence's context-free effect can be shared while its occurrence-conditioned interpretation cannot. Keep primal and response dependency guards separate. Hashes nominate equality checks when collision freedom is not guaranteed; they are not biological evidence.

Start with two worlds and a static DAG, not a general distributed persistent runtime. Measure deduplication bookkeeping and retained state. This composition connects BP repeat/edit mechanisms to GH alternative hypotheses using CE execution ownership.

## N10 — Guarded specialization that keeps the derivative honest

**E inside a proved guard; M/A for a learned/hardened substitute.** Specialize stable constants, shape, support and query type at preparation. Bind guard conditions to the program. On a guard failure execute the general route or explicitly reprepare. A parameter may be unchanged across many forwards yet still require a gradient; replacing its computation by a detached cache would be wrong. Cache a differentiable factorization/tape dependency or expose the declared constant-versus-trainable distinction.

A small typed rewrite space can search Boolean identities, matrix factorizations supplied by the model, fusions and layout moves. Restrict floating reassociation to the selected numerical policy; exact integer rules do not license arbitrary floating rearrangement. `egg` is an implementation precedent [R04], not a requirement to introduce an e-graph dependency. Monarch is relevant precedent for useful structured matrix families [R05], not proof that this programme matches its model or measurements.

## N11 — Support algebra as an executable routing representation

**E for declared finite relations; A if support is inferred or thresholded.** Compose Boolean relations with OR-of-AND, intersect typed supports, perform rank/select into compact payloads, and emit factorized groups instead of unnecessary pair expansion. For small binary relations, ordinary 0/1 FP16 multiplication with bounded FP32 path counts followed by thresholding can realize existence composition. This is arithmetic emulation with an explicit count/range policy, not binary tensor-core hardware on Volta.

Use popcount/bit operations for compact supports and genuine matrix contraction where the dimensions make it useful. Max-plus or arbitrary semirings are different mathematics; they do not automatically map to ordinary MMA. Candidate support, primal dependency, response dependency and sequence validity remain distinct universes.

## N12 — Coordinate changes and residual capacity as ordinary library operations

**E for a verified invertible map; A/M otherwise.** For q=T s, transform the law as G(q)=T F(T⁻¹q) and the readout as r'(q)=r(T⁻¹q). A local actor basis change transforms ports rather than canonicalizing all hidden columns. A quotient needs an invariant retained manifold or an explicit approximation; support similarity or one batch of equality does not suffice.

Publish the rewrite with a coherent state/parameter/readout/optimizer migration. A feed-forward residual V*tanh(Ux) with nonzero U and initially zero V preserves the initial output while allowing outgoing gradients. It is not automatically a safe recurrent-state insertion. Recurrent capacity needs a declared initial state and dynamics. This makes the earlier extract/publish/regrow idea a usable mathematical facility without requiring automatic discovery of a uniquely identifiable regulator.

## Existing unusual operators remain in scope

M12 texture atlases, M13 multi-contraction precision, E45 texture tables, E46 DP4A/butterflies, E47 instruction search and E48 residency/recomputation remain candidate providers. Their numerical/gradient semantics are not interchangeable. A software interpolation reference, a hardware texture primal and a sampled derivative texture require an explicit convention. Range-scaled hi/lo contractions do not become exact FP32 merely because four products are accumulated. Keep those distinctions in capability and result records.

## Invention acceptance

N01–N06 include executable host witnesses; N02 includes an authored SM70 kernel. The rest specify integration/search mechanisms around existing code. No new GPU benchmark result is claimed by this package. Implement at least two useful new native compositions, retain the broader portfolio and report unsupported directions without pretending the scientific frontier is finished.
