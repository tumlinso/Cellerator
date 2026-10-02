# Architectural search: possible Baseplanes

## The most useful change of question

Do not ask only, “What vector should represent this stretch of DNA?” Ask:

**What can this region answer, what can it do to incoming state, what detail can it recover, and how can it find the other regions it needs?**

A content vector is one answer. A state-transition table, conditional response operator, residual-backed summary, exact posting list, typed question or factorized relation is another. Baseplane may combine them, but the workshop should not freeze them into a universal IR before we know which combinations work.

The proposal is a portfolio of architectures, with three particularly strong organizing principles: **effects**, **refinement**, and **rendezvous**. These are not three mandatory pipeline stages. They are different ways to avoid repeatedly expanding all sequence into the same dense representation.

## 1. A region as an effect rather than a bag of features

Suppose a region implements a small state transformation T. Two adjacent regions L then R compose as:

    T_parent(s) = T_R(T_L(s)).

This preserves sequence order. It can distinguish arrangements that a histogram collapses. It also answers a new incoming-state query without re-reading the entire region, provided the chosen transformation family is expressive enough and compact enough.

### Thirty-two possible states fit the warp itself

Let lane s hold T(s), for s in 0..31. Each lane holds one entry of a complete transition function, not one base. To compose left then right, lane s needs the right table's entry at index left[s]:

    parent_s = __shfl_sync(full_mask, right_s, left_s);

The state lookup *is* the communication pattern. No divergent tree of 32 branches is necessary. The table is a piece of computation that can move upward in a hierarchy. This is a proposed mapping derived from function composition and Volta shuffle semantics, not a measured speed claim. [S02; E05]

A counted transducer extends it:

    state_parent(s) = state_R(state_L(s))
    count_parent(s) = count_L(s) + count_R(state_L(s)).

That is a second shuffle plus addition for the counter field. First/last witnesses may also compose. Full variable-length event histories generally do not fit into the same fixed-size object; ask the exact source to reconstruct them when necessary. [E06]

The intriguing possibility is that a learned local model could eventually be discretized into such a small state vocabulary. A dense learner would discover a useful operator; the cheap path would execute its compiled finite-state consequence. The vocabulary and approximation error remain scientific design choices, not facts about the genome. [S15–S16; E24, E29]

### Continuous effects that remain closed

The existing CarryFold is diagonal affine. A more expressive yet still compact family is:

    h_out[i] = d[i] * h_in[p[i]] + b[i],

where p is a permutation. For L followed by R:

    p_parent[i] = p_L[p_R[i]]
    d_parent[i] = d_R[i] * d_L[p_R[i]]
    b_parent[i] = d_R[i] * b_L[p_R[i]] + b_R[i].

Channels now move through the computation instead of evolving independently. With one coordinate per lane, permutation composition and operand gathering are shuffles. The representation is closed under mathematical composition, but floating-point evaluation is only approximately associative. Arbitrary nonlinearities and arbitrary dense mixing break this closed family; insert them as explicit boundaries, or try block-monomial effects. [E09–E12]

This can produce an interesting alternating architecture: rich local learning discovers or updates a response; a cheap compositional corridor transports state over a larger region; a later interaction triggers richer computation again. Mamba is relevant prior art for input-dependent state-space computation and hardware-aware scans, not a guarantee that any desired response admits a tiny summary. [S07]

### Why this might fail usefully

An operator table can be larger than the raw sequence of a short region. A 32-coordinate permutation/scale/offset representation is not cheap enough to attach indiscriminately to every base. Promote at a useful region scale, exploit repeated queries, compress the operator family, or discard the idea for that workload. Its value is the ability to answer *many incoming states*, not a magical compression ratio.

## 2. Coarse representations with an explicit way back

A hierarchy need not irreversibly discard everything it does not carry upward. Lifting gives a concrete construction. For an even/odd pair, with prediction P and update U:

    detail = odd - P(even)
    coarse = even + U(detail).

If the same functions and detail are retained:

    even = coarse - U(detail)
    odd = detail + P(even).

The coarse path can stay small while detail lives in a separate reservoir. Mathematical reversibility does not imply bit-exact floating reconstruction; an integer lifting variant would need explicit rounding and overflow semantics. Nor does retaining a full residual tree automatically save memory. The potential saving is that most queries need not *move or process* all detail. [S10; E13]

There are several fundamentally different choices worth implementing. Store all residuals but stream only the requested ones. Compress residuals into significance bitplanes. Retain only an exact source reference and reconstruct local detail by re-encoding. Use a learned predictor so the residual distribution changes during training. Each choice moves cost between memory, recomputation, error and query latency.

### A question should control refinement

A higher layer might ask for a motif count, a conditional response, an order-sensitive feature, or a relation score. A region can answer from a suitable summary; otherwise it asks its children or exact source. Some questions admit certified bounds. Others only have learned confidence. Those cases must be tagged differently.

The supplied scalar residual-tree fixture demonstrates a genuine bounded query: a subtree maximum safely rules out scalar values above a threshold. That certificate applies to that scalar query, not to arbitrary biological importance. The lesson is to give every proposed rejection rule a precise meaning. [E25, E39]

Precision can itself be refinable. Start with coarse latent intervals or a few bitplanes; materialize finer residuals only when a decision remains ambiguous. This differs from simply quantizing everything once: resolution becomes part of the active query. [E15]

### Learned boundaries without an annotation oracle

H-Net's learned dynamic chunking and BLT's variable patching show that segmentation need not be fixed tokenization. Baseplane can try learned boundaries, simple content-defined boundaries, multiple seam systems or a small forest of competing parses. These are alternatives, not a requirement to copy a language-model architecture. [S08–S09]

Do not equate surprise with function. A regular low-entropy region can matter to a particular question, while a surprising region can be irrelevant. Keep an exploration path and allow later evidence to reopen a coarse decision. A representation can learn from its own counterexamples: find two inputs that it collapses despite different target answers, then add a local predicate, residual or split. [E16, E37–E40]

## 3. Nonlocal context needs an actual directory

A warp can group equal values already in that warp. It cannot discover every distant genomic object with a compatible key. `match.any.sync` is explicitly warp-local. [S02]

A real nonlocal path is:

    sequence/current representation
      -> input-derived keys
      -> whole-input bucket/posting directory
      -> candidate groups, including groups larger than 32
      -> exact or richer relation evaluation
      -> source-grounded sparse messages.

Start with sort/count-prefix-fill. It makes the semantic contract visible and avoids concurrent publication races. A warp-cooperative hash table or multisplit can later change the executor without changing what constitutes complete group membership. [S13–S14; E17–E18]

Approximate sketches can nominate candidates, but candidate recall is a separate quantity from final score quality. Multi-probe keys, exact string checks and learned relation maps serve different roles. An exact compressed index can be another candidate generator, but its construction and output costs remain part of the path. [S12; E19]

Popular keys are a danger: complete pair expansion of a group of size k requires k(k-1) directed outputs. No compact directory makes that output free. Keep a factorized group, aggregate, tile the work, or declare a budgeted approximation. Do not silently truncate and then call the result global. [E17, E43]

### Invert the question instead of precomputing every feature

A more unusual design uses subscriptions. A higher-level state asks for a typed condition; the exact sequence or an appropriate summary produces answers to those requests. Equivalent predicates can be shared across requesters while destination and source restrictions stay distinct. Computation resembles batched query execution or packet routing rather than a fixed neural layer applied everywhere. [E20, E26]

That offers a route for Cellerator/GlassHelix context to shape sequence interrogation without making Baseplane own the whole biological state model. The local exchange is a source-grounded query/answer contract; the shared numerical meaning remains at its proper owner.

## 4. Tensor Cores need meaningful matrix axes

The first good tile is not an arbitrary reshape of a long vector. It is something like:

    A[object, feature] x B[feature, new_feature]
      -> transformed features for 16 promoted objects.

The rows can be related objects inside one genome or one regional computation. External batch size need not be 16. Useful work on the rows, features and packing path must still justify the tile. [S01, S03; E21]

A second tile is `Q K^T` over nominated events. A third is 16 possible entry-state vectors acted on by one regional transformation. These use Tensor Cores for different questions: feature mixing, candidate relations, and sampled regional responses. [E22, E24]

A deliberately unusual fourth option uses 0/1 matrices as tiny finite relations. Ordinary arithmetic multiplication counts two-step paths, and thresholding recovers existence. That is not native Boolean MMA, and it does not turn an FP16 Tensor Core into an arbitrary semiring processor. It is a bounded arithmetic construction whose correctness can be checked against an integer relation oracle. [S02–S03; E23]

Keep standard WMMA fragments opaque. A separate low-level experiment can investigate documented sm_70 FP16 `mma.m8n8k4` and inspect its native lowering, but should not make an empirical register mapping a universal ABI. [S02, S04–S05]

## 5. Let learning eventually become cheap execution

The blue-sky goal is not to eliminate learning. It is to discover which learned regularities can be represented more cheaply than repeated generic dense evaluation.

A small learned circuit can harden to Boolean gates. A continuous routing score can select a compact predicate bank. A repeated question can partially evaluate into a specialized program guarded by source, weight, state and query versions. A tiny offline rewrite system can search equivalent LOP3/funnel/shuffle expressions. [S15–S18; E29–E32]

The supplied synthetic learner fits eight logits and emits a LOP3 truth-table immediate. It is intentionally modest: it demonstrates the plumbing from continuous parameters to machine-level logic. It is not a genomic model, a generalization result or proof of an interpretable biological circuit.

Do not turn this into a second Cellerator compiler. The experiment is a small local lowering problem with explicit input/output semantics. If it grows into general numerical planning, record a producer-owned handoff rather than annexing that responsibility.

## 6. Share exact content, not assumed biological meaning

Repeated sequence suggests reuse. But the same substring at two loci, in two chromatin contexts or under two cellular states need not mean the same thing.

A safe computational decomposition is:

    shared exact content / context-independent local effect
      + occurrence-specific context and source map
      + state/query-dependent residual or response.

Hashing nominates equality; exact comparison or collision-safe structural equality establishes it. Cache keys declare every relevant version. Invalidation follows both value dependencies and changes in hierarchy or retrieval membership. [S11–S12, S17; E33–E36]

This creates a plausible variant engine: an immutable sequence/effect DAG plus sparse alternative worlds. Unchanged work is shared, and computation splits where an edit changes a predicate, state response, boundary or key. A local edit can still have a large global consequence, so the dirty cone is an observed dependency structure rather than a constant-time guarantee. [E34, E44]

## 7. Moonshots that should not be discarded before implementation

**Boundary-port regions:** compress a synthetic regional system to the ways its interior can influence a small interface. Static condensation and reduced-basis methods motivate the algebra. Whether useful biological ports can be learned is the new question, not something those methods already prove. [S19; E41]

**Coarse corrections rather than one-way pooling:** pass a compact residual upward, solve a coarse consistency problem and send corrections down only where needed. This makes hierarchy a mechanism for revision, not just compression. A neural version has no automatic convergence guarantee. [E42]

**Hyperedge factors rather than all-pairs graphs:** carry partially filled typed roles, intersect postings and evaluate only surviving combinations. This can express a candidate requiring several conditions jointly, while preserving factorized groups instead of expanding every pair. [E43]

**Several possible worlds in one warp or tile:** share immutable context across variant/state hypotheses, then branch in representation through support masks. Merging worlds requires query-relative equivalence, not merely similar vectors. [E44]

**Texture hardware as a response surrogate:** a small learned response surface can become an explicit floating texture lookup. Finite interpolation precision makes this a numerical approximation, not an exact sequence predicate. It is worth implementing precisely because it uses a rarely considered part of the available machine. [S03; E45]

**Instruction composition as a search space:** synthesize tiny exact circuits or permutations, reject wrong candidates with bounded truth tables, compile survivors and inspect SASS. The target is not simply shorter source code; it is a representation that fits the actual register and movement structure. [S02, S04–S05; E47]

## 8. A loose common substrate, not a mandatory universal object

Share only what several experiments actually need: source references, exact validity access, explicit buffer/counter policy, small fixture generators, a record of semantic mode and dependency versions, and simple compilation/run entry points. A common header with a single huge carrier layout would defeat the purpose.

A local exchange *may* describe source support, operator/representation kind, level, child or residual references, and a compact numerical sidecar. Those fields can live in structure-of-arrays layouts or family-specific records. A base is not required to have a carrier. Public Cellerator identity and numerical contracts are not invented here.

Make a composition demonstrate at least one substantive cross-family interaction. Merely launching every experimental kernel in a list is not an architecture. Ten concrete combinations are specified in `machine/compositions.json`; several can be built without making any one of them “the” Baseplane.

## 9. The cost hypothesis, stated without magic

For a particular workload, a useful decomposition is:

    total work = exact input/build work
               + active representation work
               + candidate discovery and routing
               + rich local computation
               + refinement/replay
               + requested output.

For an arbitrary unseen sequence, exact comprehensive inspection cannot generally be free or sublinear in the input it must read. The opportunity is to avoid repeatedly expanding all positions into expensive context, and to amortize discovered structure across appropriate queries or state updates.

A hierarchy can lose if it emits too many carriers, builds too much metadata, creates broad posting lists, repeatedly reopens detail, or spends more on packing than on useful arithmetic. These are later measurements, but the designs should expose the counters now.

The workshop's success is not a finalized architecture. It is a set of enough concrete, intelligible alternatives that the next scientific and engineering campaign can compare real mechanisms rather than slogans.
