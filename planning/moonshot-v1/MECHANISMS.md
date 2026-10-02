# Mechanism cards

These are implementation directions, not a compulsory single architecture. **E** means equivalent lowering for a supplied expression, **M** a model-family choice, **A** an approximation, **P** a prototype policy. None implies biological identification. Filenames refer to `payload/cellerator/` unless specified. Source labels resolve in [research/SOURCES.md](research/SOURCES.md).

## M01 — Matrix patches: put the algebra on both sides of the state

**Class:** M; E after the model declares the factorization. **First wave.**

Gather a unique ordered coordinate list into `X[16,16]`; define `Y=tanh(LX)R`, or use the linear member `LXR`. Each patch has its own `L,R`. A permutation or sparse exchange between layers makes later patches see different neighborhoods. There is no requirement to interpret patch columns as the same hidden regulator across actors.

**Algebra:** for `T=LX`, `V=tanh(T)` and output cotangent `G`,
`dR=V^T G`, `dT=(G R^T) .* (1-V^2)`, `dL=dT X^T`, `dX=L^T dT`. JVP: `dT=dL X+L dX`, `dY=((1-V^2).*dT)R+V dR`. Preserve these intermediate values or recompute from the recorded primal.

**Layout/schedule:** patch-major half operands, FP32 accumulator/output. Four warps per CTA; one patch per warp. First WMMA, store into that warp's shared slice, apply nonlinearity/convert, second WMMA. The seed intentionally uses shared memory rather than inventing a portable mapping of WMMA fragment registers. Later specialize a verified SM70 fragment-to-fragment path if it actually simplifies the schedule [R3,R16].

**Source:** `kernels/patch16_sm70.cu`, `reference/moonref/operators.py`, GH `actor_fields.py:MatrixPatchField`. The CUDA nonlinear intermediate is rounded to half; the float64 reference is the mathematical family, not a bitwise CUDA oracle. Add a stored-half reference when binding the GPU path.

**Cost/failure:** broadens performance options but restricts the model versus arbitrary 256x256 coefficients. Small patches can be launch-bound; use one launch for many patches of one cell. Do not equate number of patches with cell batch. Aliased outputs, duplicate gather IDs and scatter reductions need declared semantics.

## M02 — Matrix polynomial reactions

**Class:** M; E for the declared polynomial. **First/second wave.**

Define `F(X)=LX+XR+XMX`. This gives state-dependent quadratic interaction through matrix algebra, without storing a full quadratic coefficient tensor. It is more expressive than a fixed linear patch but still a restricted family. Add several differently permuted terms if needed.

**JVP:** `LP+PR+PMX+XMP`. **VJP:** `L^T G+G R^T+G(MX)^T+(XM)^T G`. Parameter derivatives are `dL=GX^T`, `dR=X^T G`, `dM=X^T G X^T` for the stated square law. Shared intermediates `MX` and `XM` can serve forward and selected responses under the same epoch.

**Implementation:** patch descriptors identify factor sets, not per-edge callbacks. Keep each product chain owned by a warp/CTA and assemble once. Use FP32 around gates and nonlinear terms. A general polynomial expansion is not required; start with this exact small vocabulary.

**Source:** CPU forward/JVP/VJP in `operators.py`, derivative smoke included. Native GPU composition and parameter gradients remain to implement. A matrix commutator `LX-XL` is another useful experimental member for oscillatory latent transport, but its invariants are properties of that family, not proof of biological conservation.

## M03 — Exact local linear flows

**Class:** M/E. **High-value trajectory experiment.**

For `dot X=LX+XR`, execute `E_L X E_R`, where `E_L=exp(dt L)`, `E_R=exp(dt R)`. Cache only under `(definition,parameter_generation,dt,precision)`. Same dt and same factors can reuse exponentials across many updates; changed dt or factors requires refresh.

**Forward-state derivative:** `P_next=E_L P E_R`. **State VJP:** `G_prev=E_L^T G E_R^T`. Gradients for `L,R` require differentiation of matrix exponential, not the derivative formula of an ordinary trainable `E_L,E_R` unless those are actually the model parameters. Torch `matrix_exp` is the delivered reference route. Never detach this dependence during learning.

**Scheduling:** a prepared exact-flow operation can replace a sequence of explicit steps for this local law. Nonlinear/external coupling needs an explicitly chosen composition or splitting method. The flow block is not a universal solver. Do not claim an arbitrary `LXR` vector field has this solution.

**Source:** `operators.py:sylvester_flow`, GH `actor_fields.py:SylvesterFlow`; CPU gradient witness run. Cache/scheduler integration remains to implement.

## M04 — Four microcontractions per Volta warp

**Class:** E. **First wave.**

Queue jobs `A_j[8,4] B_j[4,8] -> C_j[8,8]`. Pack four jobs into one warp using PTX `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`. Iterate K in four-wide pieces when necessary. Jobs may have different coefficient values [R2].

For lane `l`, job-in-warp is `(l>>2)&3`; A row and B column are `(l&3)+(l>=16?4:0)`. Each lane supplies four half A and four half B values, packed in two 32-bit registers per operand. The seed follows the documented FP32 accumulator mapping and provides an unrun SM70 smoke test. The four jobs are an instruction-level grouping, not independently synchronized subwarps.

**Use:** small port-response panels, low-rank messages with true width, matrix-polynomial terms, different local learned operators. If a job has one useful RHS column, disclose unused arithmetic; grouping four jobs does not magically supply eight useful columns per job.

**Source:** `kernels/quad_mma_sm70.cu`, `tests/gpu_smoke.cu`. CUDA12, target70. No vendor peak-throughput argument substitutes for compiling this unusual layout correctly.

## M05 — Private hidden bases, shared message ports

**Class:** M/E for the chosen interface. **First wave.**

Each actor has private state `h_i[H_i]`. Compute `m_i=E_i h_i`, `z_i=sum_j A_ij m_j`, `dh_i=phi_i(h_i)+D_i z_i`. Width `P` is the message-interface size, not an identified biological hidden dimension. Ports can be typed by operation/role; different families need not share one universal port space.

**Backward:** `bar_z_i=D_i^T bar_dh_i`; accumulate `bar_m_j += A_ij bar_z_i`; `bar_h_j += E_j^T bar_m_j`; `bar_A_ij=bar_z_i^T m_j`, with local-map gradients formed as the corresponding outer products. Local law derivatives are additional. Repeated edge/parameter IDs retain their declared multiplicity.

**Packing:** `M[actors,P]` supports sparse relation application with real single-cell width. `E_i,D_i` vary per actor; choose direct vector kernels, micro-MMAs where genuine panel width exists, or an explicitly learned map palette. A palette ties parameters and changes the hypothesis family; shared source code alone does not.

**Source:** `operators.py:local_port_transport/reframe_ports`, GH `ActorPortField`. The independent-basis witness verifies the linear interface. Do not canonicalize private embedding columns just to pack messages.

## M06 — Process packets and warp-local fanout

**Class:** E. **First/second wave.**

Make each lane own a local process, not necessarily one output state variable. For a two-input example, `r=k*x[a]*x[b]`; JVP is `k*(v[a]*x[b]+x[a]*v[b])`. Group processes by opcode/arity, then by shared source IDs. `match_any_sync` identifies equal gather indices and a selected lane loads once before `shfl_sync` shares the result [R3]. This only shares the input load; coefficients remain distinct.

The delivered kernel handles partial warp packets with a sentinel and keeps collective participation uniform. It writes one result per process. A second prepared assembly uses output ownership/reverse incidence. Do not use process output as an excuse for unsynchronized writes into a simultaneously read cell state.

For longer products, use prefix/suffix derivatives rather than dividing by an argument. Repeated inputs retain multiplicity. The current indexed mechanism implementation is a direct native integration point.

**Source:** `kernels/product2_jvp_sm70.cu`, `operators.py:product_value_jvp`. Shared-read benefit can disappear if the cache already serves loads cheaply; this remains an experimental candidate, not a new default.

## M07 — Topology as bit operations and lane schedules

**Class:** E for representation; A if support is inferred by thresholding. **First/second wave.**

Give each computational object separate tagged input/output/context footprints. Use exact bitset intersection/popcount, or MinHash/LSH to shortlist groups. Pack shared producers/consumers and compatible opcodes; use an execution-cost proxy including padding, number of unique loads, output reductions, and reuse.

For a 32-lane neighborhood, a support word can replace a list of local indices. Lane membership is a bit test. Packed payload offset is `popc(mask & ((1u<<lane)-1))` with appropriate handling of the full word and lane31. `ffs` selects the next entry. Forward and reverse incidence may use different orders.

Support words describe **declared dependency or structural existence**, not whether a float happens to be zero. Low gradient importance is not structural absence. The prototype's Jaccard is a co-placement signal; it never merges names. Small footprints often need direct comparison rather than sketches.

**Source:** `reference/moonref/packing.py`; adapt to existing geometry compiler rather than replacing it. A more speculative large-shortlist path converts boolean incidence to 0/1 half panels and obtains intersections via a Gram product. FP32 integer accumulation is exact only within its representable integer range; chunks and normalization must be explicit. Volta has no binary Tensor Core MMA. Compare against popcount rather than assuming Tensor Cores win [R2].

## M08 — Extract, publish, and regrow without a dead branch

**Class:** P, with E only for a supplied verified coordinate/refactoring map. **Core learning loop.**

Promotion request fields: old model epoch; ordered old/new coordinate IDs; extraction/state initialization map; new law/readout declarations; coefficient correspondence; optimizer migration/reset policy; physical projection rebuild list; evidence/scope label. It may name an anonymous meta-actor. Slots have incarnation IDs so an old tape cannot mistake a reused address for the old variable.

For invertible `q=T s`, `G(q)=T F(T^-1 q)` and `r_new(q)=r_old(T^-1 q)`. For a quotient, `s=E q` must describe an invariant manifold and `PE=I`. Equality in one batch is not enough. A physical replica gradient reduces by `E^T`, while a state-copy operation uses `E`; do not average gradients merely because values are replicated.

Drain or retain valid old tapes; publish new structure atomically through current ownership machinery. Reset affected Adam moments for a basis-changing prototype unless a mathematically suitable explicit migration is supplied. A permutation is much simpler than a dense basis change.

Recycle residual capacity using `V tanh(Ux)` with nonzero seeded U and initially zero V. Output initially unchanged; V generally receives nonzero gradient. Clear moments only for recycled coordinates. Subsequent fitting can repurpose them. Total model size need not stay constant: a promoted actor also consumes state and parameters.

**Source:** `reference/moonref/promotion.py`, GH `RegrowthBranch`. Seed includes a linear exact-refactoring witness, not automatic nonlinear scientific extraction. Related precedents: function-preserving morphism and dynamic sparse training [R9–R11].

## M09 — Response packets and saved gradient contractions

**Class:** E for genuinely requested derivatives. **Second wave.**

For a common linear map, pack `[h, v1,...,vp]` only if those primal/direction quantities are actually needed. Apply the map to all columns; continue nonlinear JVPs through their chain rule. An activation is not applied independently to tangent columns as if they were cells.

For `Y=LXR`, compute `LP R`; for nonlinear patches reuse the recorded `1-tanh(LX)^2` factor. For product processes use M06. State and parameter directions can be combined, but their labels/scales remain explicit.

Weight gradients accumulated across recorded rollout positions can form a GEMM after the relevant adjoints exist: `dW=sum_t g_t x_t^T`. This does not parallelize causally dependent forward time steps. Keep one weight generation or explicitly separated generation contributions. Never average gradients as an undocumented normalization.

**Source:** CPU patch JVP/VJP and product references. Native forward/backward adapters are CE-MOON-DIFF work, reusing saved-primal/tape contracts. NF1 capability bits must say what actually exists; a CUDA forward kernel does not grant VJP/JVP support automatically.

## M10 — Delta ledger with remembered transmission state

**Class:** E in real arithmetic for a fixed linear map and all changes; A with thresholds or finite precision reuse. **Second wave.**

Keep `x_sent` and `y=W x_sent`. Find changed coordinates using `abs(x-x_sent)>tau`. Apply `W[:,changed]*(x-x_sent)[changed]`; publish those sent values only after the update. This preserves subthreshold accumulation until it becomes worth transmitting. A threshold against the previous physical time step loses gradual changes.

A conservative instantaneous discrepancy is bounded componentwise by `abs(W)@abs(x-x_sent)`, plus floating-point recurrence error. No trajectory bound is implied. Refresh after parameter-generation changes and periodically if floating-point drift matters. For nonlinear operators use cached linear pieces or an explicit Taylor approximation, never call the whole update exact.

Build separate forward and response invalidation sets. A dormant primal contribution can still have nonzero derivative. Start from a fixed linear layer and do not require a global asynchronous simulator.

**Source:** `reference/moonref/delta.py`; unit test includes repeated small untransmitted changes and parameter invalidation. Related recurrent delta-computation precedent [R12].

## M11 — Low-rank feedback islands

**Class:** E for the supplied linear solve; M/A if the Jacobian is approximated. **Second wave.**

For `J=D+UV^T`, let `B=I-dt D`. Solve

`(I-dt J)^-1 b = B^-1 b + B^-1 dt U (I-V^T B^-1 dt U)^-1 V^T B^-1 b`.

The large operation becomes diagonal work, narrow contractions, and a small dense solve. This is useful where a coupled feedback region actually admits low-rank structure. It can complement matrix-native forward math by reducing integration work. The reduced solve needs conditioning/pivot policy; do not implement an unstable inverse just to invoke a Tensor Core.

JVP/VJP for the solve use the usual implicit linear-system derivative and transposed solve, retaining the same matrix generation. Do not use the derivative of a converged mathematical solve when a deliberately truncated iteration is the declared differentiated object without stating the distinction.

**Source:** `operators.py:low_rank_implicit_solve`, float64 equivalence witness. Native integration and a GH solver decision remain future implementation.

## M12 — A rate-law texture atlas

**Class:** A. **Deliberately unusual optional experiment.**

For many repeated two-input bounded rate laws, tabulate each law on a grid in an explicitly chosen coordinate system (often logarithmic only where positivity permits). Store an immutable texture layer per law/palette entry. `tex2DLayered` performs the interpolated lookup; exact domain scaling maps sample nodes to texel centers. Hardware supplies low-precision interpolation [R15]. This trades arithmetic for a cache-backed fixed-function unit rather than more Tensor-Core instructions.

Keep parameter generation in the atlas identity. Changed trainable laws require rebuild; this is likelier to help inference or slowly changing parameters. Values outside the prepared domain use an explicit fallback, not silent clamping that changes the biology.

For derivatives, either evaluate a specified continuous interpolation in software from four samples, or declare a surrogate. A sampled derivative texture is not automatically the derivative of the hardware-interpolated primal. Texture coordinate rounding makes exact machine-level differentiation a separate question.

**Source:** design card only, no delivered CUDA implementation. Prototype one saturating law with a tiny visual/numerical error table. Do not demand a universal nonlinear-table compiler.

## M13 — Precision as multiple contractions

**Class:** A. **Optional numerical experiment.**

Represent an operand as `A_hi + alpha A_lo` and similarly B, each component half-stored. Compute three or four component products with FP32 accumulation. Three omits `alpha beta A_lo B_lo`; four retains it but still has storage/arithmetic rounding. Neither is universally exact FP32 GEMM.

A block power-of-two scale can protect range and preserve cheap rescaling. Check finite conversion/underflow policy. Keep FP32 canonical state or parameter owners where required, and bind derived planes by generation. Do not quietly move optimizer masters to half.

This can make Tensor-Core-rich small nonlinear/derivative contractions useful in a wider numerical regime, at extra work. It should remain opt-in until a model-specific tolerance makes sense. Published high-accuracy Tensor-Core constructions are precedents, largely on later hardware, not V100 performance evidence [R14].

## M14 — Resident regions and sparse mailboxes

**Class:** E with fixed snapshot semantics; A if scheduling changes the numerical method. **Integration experiment.**

Prepare a dependency DAG of regions. Each reads an immutable step snapshot, computes local intermediates, and writes owned contributions. Use kernel boundaries and the existing program for global ordering initially. Within a CTA, stage immutable source panels and use explicit synchronization; keep accumulators live through local chains. End with an assembly/step publication.

For sparse activity, a compact list contains region IDs, not individual divergent lane decisions. Inactive regions retain a wake-up dependency. Different warp predicates are legal only when subsequent synchronization allows them. Avoid a software global barrier inside an ordinary oversubscribed kernel; waiting blocks can prevent unscheduled producers from ever running.

A later cooperative persistent implementation must prove residency/launch feasibility and backpressure. NVLink can serve replicated read-mostly regions or separate candidates; do not shard a small tightly coupled cell across devices by default. Discover live topology before choosing communication assumptions. The prototype need not use every GPU to be worthwhile.

The boldest extension is a compiled region whose lane schedule *is* the sparse graph: constant masks, register shuffles, fused local nonlinear laws and MMA islands. Generate that inside Cellerator's existing compiler/candidate system, not a GH interpreter.
