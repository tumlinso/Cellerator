# Experiment catalogue

Research ownership is specified in each card and `machine/experiments.json`. Cellerator numerical providers may own the learned mechanism from inception; the listed Baseplane families retain sequence-grounded consumers. See [paired scope](07_CELLERATOR_RESEARCH_SCOPE.md).

48 concrete directions grouped into 12 architectural or machine-mechanism families. This is not a claim of 48 implementations or 48 independent scientific breakthroughs. Family-level TODOs keep execution coarse-grained. Each card defines what is different, how to begin, the machine mapping, and a semantic limit.

## Masks as executable geometry

**[E01: LUT3 circuit tiles](../experiments/E01.md).** Can a learned local decision become a few truth-table instructions rather than a feature tensor? Represent 32 positions per Boolean plane. Build a typed three-input gate DAG over exact bases, validity, strand and prior threshold masks; emit several planes for later learned mixing. Compile fixed truth tables into LOP3 immediates. Keep weights/truth-table provenance beside the circuit.

**[E02: Carry-plane evidence counters](../experiments/E02.md).** Can many predicates accumulate graded evidence without unpacking each genomic position? Maintain bit-sliced counters using XOR for sum and AND for carry. Add motif supports and threshold the counter planes with Boolean comparison. Explore weighted two-bit votes or saturating counters as separate semantics, not accidental wraparound.

**[E03: Strand-covariant local grammar](../experiments/E03.md).** Can orientation and relative spacing be data transformations rather than duplicate branching kernels? Build forward/reverse mask operators with explicit coordinate transforms. Compose spacing, adjacency, exclusions and run boundaries using cross-word funnels. Canonicalize a grammar with a strand tag, but retain both anchors when the requested duplicate policy requires them.

**[E04: Mask-index query algebra](../experiments/E04.md).** Can a support mask be simultaneously a predicate, a local index and a routing instruction? Pair support with rank/select. Intersect two questions in mask space, use rank to address compact descriptors, and select to return to original coordinates. Compose local support maps rather than materializing dense position IDs at each stage.

## Finite-state effect machines

**[E05: A warp is a transition function](../experiments/E05.md).** Can one warp represent what a region does for every possible entry state? Let lane s hold T(s) for a 32-state machine. A concatenated region stores R(L(s)); fetching lane L(s) from the right table is a shuffle. Build a hierarchy of these maps, with explicit state vocabulary and reset semantics. Query a root map without traversing every base.

**[E06: Counted and witnessed transducers](../experiments/E06.md).** Can an effect summary retain enough evidence to explain its outputs? Augment each possible entry state with an emitted-count monoid and an optional first/last witness. Compose counts as cL(s)+cR(TL(s)); transform witnesses into parent coordinates. Defer full event reconstruction to a source revisit when bounded summaries cannot hold it.

**[E07: Bounded stack grammar tiles](../experiments/E07.md).** Can local syntax be processed as a compact register machine rather than a large token model? Implement a tiny bounded push/pop transducer for nested synthetic grammar or paired sequence evidence. Store net depth, prefix underflow requirements and a short unmatched boundary stack; compose tiles when the bounded summary suffices, otherwise request detailed replay.

**[E08: Small-state weighted alternatives](../experiments/E08.md).** Can a region carry several possible state transitions instead of a single deterministic interpretation? Use tiny weighted automata with sparse destination masks and scores. Try probability, max-plus and Boolean relation versions as different algebras. Compose sparse rows and prune only under a declared approximation policy; retain alternatives for later queries.

## Continuous response algebras

**[E09: Permutation-diagonal affine effects](../experiments/E09.md).** Can CarryFold mix channels while keeping a small composition algebra? Represent h_out[i]=d[i]*h_in[p[i]]+b[i]. Compose permutations, scales and offsets using the derived monomial-affine rule. This introduces channel re-routing while remaining closed; apply nonlinearities only at explicitly represented breaks in the algebra.

**[E10: Block-monomial response maps](../experiments/E10.md).** Can a middle ground between diagonal scans and dense state matrices preserve useful mixing? Partition latent state into fixed small blocks. Each output block selects one input block, then applies a small dense matrix and offset. Compose block permutations and block matrices without densifying the block graph. Permit occasional explicit repartition steps outside the closed segment.

**[E11: Regional polynomial jets](../experiments/E11.md).** Can a region cache its local response to a small change in cellular state? Store a value, small Jacobian and optional second-order terms around an explicit expansion point. Compose local response jets and route queries inside a trust region through the approximation; outside it, refine/recompute and update the expansion point.

**[E12: Bidirectional effect checkpoints](../experiments/E12.md).** Can coarse sequence context be available from either side without materializing every fine latent state? Build prefix/suffix summaries over bounded effect families. Keep coarse checkpoints and reconstruct only requested interior states by applying local effects. Explore bidirectional feature fusion after reconstruction rather than forcing both directions into one incompatible algebra.

## Adaptive and refinable hierarchy

**[E13: Reversible lifting with residual reservoirs](../experiments/E13.md).** Can expensive context travel upward while recoverable detail stays off the hot path? Split neighboring carriers into coarse and detail through predict/update lifting. Put detail in a separate reservoir, optionally compressed or stored by source reference. Higher layers read coarse values first and selectively pull residuals when their question requires them.

**[E14: Learned dynamic chunk boundaries](../experiments/E14.md).** Can the hierarchy be learned during encoding rather than supplied by annotation? Use a small context-dependent boundary scorer, compact boundary carriers and gather an inverse map for dechunking. Train the scorer with a bounded synthetic sequence task and a compute-budget penalty; compare against fixed windows and a cheap surprise-based route.

**[E15: Precision as a refinable resource](../experiments/E15.md).** Can latent information remain as coarse bits until a query needs more precision? Encode selected latent coordinates as sign/exponent class plus successive residual planes. Carry a conservative interval where possible; evaluate easy decisions with the first planes and fetch additional planes only for ambiguous comparisons.

**[E16: Content-defined seam forests](../experiments/E16.md).** Can hierarchy boundaries remain useful under sequence edits without becoming biological assertions? Choose chunk boundaries from rolling content fingerprints, then maintain a second offset or learned segmentation as an alternative. Store exact source spans and combine answers from the two seam systems when a question straddles one boundary.

## Real nonlocal candidate discovery

**[E17: Whole-directory rendezvous](../experiments/E17.md).** Can distant sequence-derived objects actually find each other without oracle-assembled packets? Generate keys from current carriers, build a whole-input directory by sort or count-prefix-fill, and expose complete posting ranges. Process groups larger than a warp in tiled loops. Keep coordinates and candidate reasons; separate key equality from final relation verification.

**[E18: Warp-cooperative posting hash](../experiments/E18.md).** Can the directory itself follow the geometry of a warp? Build a bulk, immutable hash table whose buckets hold a small row of keys and posting descriptors. Lanes inspect a bucket cooperatively and vote on the matching slot; overflow chains or a second hash choice retain all entries. Add concurrent updates only as a later separate variation.

**[E19: Approximate nomination, exact verification](../experiments/E19.md).** Can a cheap sketch send the right objects toward a precise comparison? Create compact multi-probe keys from learned low-dimensional coordinates, motif supports or sequence sketches. Union candidates from multiple directories, deduplicate, then run exact or richer learned comparisons. Record candidate misses separately from comparison correctness.

**[E20: Subscriptions instead of exhaustive features](../experiments/E20.md).** Can higher-level questions cause the right sequence evidence to be produced? Represent a subscription as a typed predicate, region/context restriction and destination ID. Batch equivalent subscriptions, evaluate them against exact sequence or summaries, and route compact answers to the requesting higher-level state. Share repeated predicates across queries.

## Meaningful Tensor Core axes

**[E21: Object-by-feature Tensor Core tiles](../experiments/E21.md).** What biologically meaningful work fills both matrix axes inside a single sequence? Pack 16 promoted objects as rows and 16 local features as columns, multiply by a learned feature transform, and retain the object-to-source map. The rows come from one region or compatible event cohort, not necessarily 16 unrelated cells or genomes.

**[E22: Relation tiles over promoted events](../experiments/E22.md).** Can Tensor Cores help decide which already nominated objects should communicate? Given 16 event descriptors, compute a small relation score tile such as Q K^T, apply explicit self/strand/support masks, then compact selected relations. Build Q and K with role-specific maps when relations are directed.

**[E23: Arithmetic finite-relation composition](../experiments/E23.md).** Can an FP16 Tensor Core tile serve as a small finite-state relation engine? Encode 16-state adjacency relations as 0/1 FP16 matrices, compute ordinary FP32-accumulated products, then threshold counts to obtain existence of two-step paths. Test bounded path-count and weighted variants separately.

**[E24: Tensor tiles of possible incoming states](../experiments/E24.md).** Can dense arithmetic cheaply evaluate several possible interpretations of the same region? Place 16 candidate entry-state vectors in tile rows and apply a shared local feature transform. Treat the outputs as a sampled regional response table; a later query selects/interpolates or refines the relevant state instead of re-running an expensive local block.

## Question-driven execution

**[E25: Query-pull refinement](../experiments/E25.md).** Can expensive features be computed because another layer asks, rather than because every base passes through the same network? Send a question plus current tolerance/state version downward. A region returns an answer from its summary, a certified bound, or a request for child detail. Batch unresolved requests by operator and depth; produce only the richer floating objects required by the active questions.

**[E26: Opcode-cohort wave execution](../experiments/E26.md).** Can information choose the next operator without forcing each lane through a different branch tree? Carriers contain a bounded opcode and operands. Partition them into cohorts, execute uniform operator kernels and emit the next wave. Compare ordinary kernel waves with a bounded resident CTA interpreter whose whole warp changes opcode together.

**[E27: Compute auctions with debt](../experiments/E27.md).** Can limited representational bandwidth be allocated without permanently ignoring quiet sequence? Assign carriers a query-value estimate, cost estimate and starvation debt. Use small discrete priority buckets instead of a full sort. Spend a fixed wave budget on high-value work while reserving exploration and aging deferred carriers.

**[E28: Sparse multiscale wavefront](../experiments/E28.md).** Can regional interpretation grow through local interactions instead of a fixed deep stack? Treat active regions as cells with small state and pending messages. Apply bounded local update rounds, promote stable aggregates, and wake fine regions when incoming long-range evidence changes their context. Distinguish synchronous Jacobi-style waves from asynchronous semantics.

## Learning that becomes cheap machinery

**[E29: Continuous gates hardened to LOP3](../experiments/E29.md).** Can useful local learning eventually become instruction-level structure? Learn a small Boolean network through relaxed gates or a small truth-table model, discretize it and emit native constant-LUT kernels. Keep the floating teacher and hard circuit separately versioned, and compare their decisions before reusing the circuit as a route.

**[E30: State-conditioned predicate banks](../experiments/E30.md).** Can a fixed set of cheap predicates adapt to the current cell without scanning with a large neural model? Learn a small state-to-bank selection or mixture over reusable sequence predicate circuits. The current state selects compatible banks or thresholds; predicates remain exact conditional questions, while bank choice is a learned hypothesis.

**[E31: Typed equality-saturation microcompiler](../experiments/E31.md).** Can we search for elegant instruction compositions rather than hand-picking one expression? Build a tiny offline rewrite space for mask DAGs, affine maps and layout transforms. Preserve types, coordinate domains and numerical modes. Extract alternatives minimizing an explicit estimated combination of instructions, live registers and movement; later inspect emitted SASS.

**[E32: Guarded partial evaluation](../experiments/E32.md).** Can repeated questions collapse into reusable cheap programs? Specialize a regional computation for stable motif parameters, query type or state subset. Record guards and versions. On guard failure fall back to the unspecialized path, and consider re-specialization only after repeated reuse.

## Reuse under context and sequence changes

**[E33: Repeat quotient plus contextual residual](../experiments/E33.md).** Can identical sequence reuse work without pretending identical sequence has identical function? Intern verified exact substrings or hierarchical child tuples. Share their context-independent sequence effects, then attach per-occurrence context/state residuals and source maps. Explore a DAG whose nodes represent shared content and whose edges retain occurrence context.

**[E34: Variant-driven dirty cones](../experiments/E34.md).** Can a small sequence edit change only the computation it actually affects? Maintain reverse dependencies from exact chunks to effects, partitions, posting memberships and query answers. On an edit, invalidate the affected value cone and structural memberships, then recompute to a fixed point or bounded wave frontier.

**[E35: Query-support memoization](../experiments/E35.md).** Can a remembered answer declare exactly what changes would invalidate it? Cache answers with their query, sequence, weight, cell-state and representation versions plus an exact or conservative dependency support set. Reuse only when both identity and support checks permit it. Distinguish result caches from cached computation plans.

**[E36: Grammar-compressed effect reuse](../experiments/E36.md).** Can repeated sequence phrases share their compositional effects at several scales? Infer a small repeated-phrase grammar or child-pair dictionary, then cache E05/E09 effects on grammar nodes. Derive per-occurrence source spans separately. Compare a static grammar pass with incremental chunk-pair interning rather than requiring a whole-genome parse first.

## Multiple representations and interpretations

**[E37: Competing parse forests](../experiments/E37.md).** Can the encoder postpone a bad boundary decision instead of irreversibly collapsing alternatives? Keep a small beam of alternative segmentations or local grammar states, sharing common prefixes and exact source references. Higher-level queries prune or refine alternatives; use a bounded packed forest rather than copying the entire sequence representation.

**[E38: Role-bound binary routing sketches](../experiments/E38.md).** Can order and role travel cheaply in a bit vector before rich floating computation? Assign codes to local objects, bind role tags with XOR and position-dependent permutation, and bundle into a sketch with explicit collision risk. Use the sketch for nomination only; retain exact supports and a richer floating sidecar for final decisions.

**[E39: Question-specific summary portfolios](../experiments/E39.md).** Can different future questions receive different compact summaries of the same region? Maintain a portfolio: counts for abundance questions, effects for state propagation, extrema/bounds for pruning and small learned vectors for similarity. Queries select the representation they need; unsupported questions cause source replay rather than forced use of one universal vector.

**[E40: Counterexample-guided resolution](../experiments/E40.md).** Can failed questions teach the hierarchy where it was too coarse? Search for query pairs or sequence perturbations whose different target answers collapse to the same coarse representation. Add a residual channel, split rule or predicate only at those failure sites. Keep random exploration so current queries do not define the entire genome forever.

## Ports, factors and counterfactual worlds

**[E41: Boundary-port response operators](../experiments/E41.md).** Can a region expose a few ways it can interact with the rest of the system instead of all its interior state? Give a synthetic regional system a small set of ports and eliminate internal variables to form a response map. Compose neighboring or related regions through compatible ports; reconstruct interior detail only when asked. Then explore learning ports rather than fixing them by genomic distance.

**[E42: Learned multigrid correction](../experiments/E42.md).** Can coarse context repair fine representations rather than merely summarize them? Construct restriction and prolongation maps between sequence-grounded levels. Run cheap coarse corrections to a local consistency objective, then refine the residual at selected fine regions. Test fixed linear maps first and learned maps as a separate experiment.

**[E43: Hyperedge rendezvous and factor joins](../experiments/E43.md).** Can meaningful multi-object combinations be found without generating every pair first? Represent a candidate mechanism as a small factor with typed roles. Use inverted postings and mask intersections to fill compatible roles, then evaluate only surviving tuples. Preserve a factorized relation when expanding every hyperedge would be wasteful.

**[E44: Counterfactual packet worlds](../experiments/E44.md).** Can alternative sequences or cellular conditions share almost all their execution? Carry a world ID and a sparse delta over an immutable sequence/effect DAG. Share unchanged computation, split only when a predicate or state response diverges, and merge worlds when their guarded execution states become equivalent under a declared query.

## Unconventional Volta execution paths

**[E45: Texture-unit response tables](../experiments/E45.md).** Can the texture hardware be a useful cheap surrogate for a small learned response surface? Tabulate a two-dimensional learned response and evaluate it through an explicitly configured floating texture object. Compare nearest, interpolated and ordinary arithmetic lookup, with coordinate and boundary conventions fixed.

**[E46: Byte dots and butterfly features](../experiments/E46.md).** Can small floating-like mixing be replaced by structured integer/register operations where appropriate? Try learned or fixed butterfly transforms for order-sensitive feature mixing, and independently try signed-byte dot products for tiny routing scores. Preserve scale/zero-point metadata and compare arithmetic meaning before combining them.

**[E47: Instruction-composition search](../experiments/E47.md).** Can Baseplane discover a good machine idiom instead of merely selecting a library routine? Enumerate bounded straight-line candidates for bit packing, rank, base permutation or state-map composition. Reject candidates by tiny exhaustive examples, compile survivors and inspect instruction count/register footprint. Preserve several alternatives without running a timing tournament now.

**[E48: Residency and recomputation choreography](../experiments/E48.md).** Can information movement become part of the representation rather than an afterthought? For a fixed semantic operator, create register-recomputed, shared-staged and global-materialized variants. Fuse cheap sequence predicates into the consumer when this avoids a full plane stream; keep a resident plane form when repeated questions justify it.
