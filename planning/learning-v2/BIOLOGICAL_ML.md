# Deliberate biology-purpose ML design

**Status:** worked research/architecture proposal and exact algebraic reference, not a claim about the actual reprogramming mechanism or a new discovery. Select/refine one construction; do not implement this entire shortlist.

## What makes an addition Cellerator-like?

Start from the scientific learning problem and ask which stable biological structure makes its computation different. Write down the modeling restriction, trainable quantities, observation meaning, mathematical operation and derivative support. Then identify what can be shared, factored, packed or avoided *without changing that model*. Compare the exact same function in a straightforward implementation and the proposed realization.

A plausible biological interpretation does not license approximate arithmetic, pruning, aggregation or a claim of causality. A performance profile may propose a layout; only mathematical equivalence or a declared scientific approximation licenses it. Biological priors belong to GH/experiment assumptions; CE receives the structured operation they justify.

A useful decision record can be short: **question → assumption → expression → unknowns/gradients → reusable structure → native/framework ownership → reference → counterexample → measured disposition**. Expensive decisions deserve this content, not a fixed number of meetings or documents.

## Candidate A: shared regulatory support with context-dependent activity

This is the provisional lowest-cost first construction, motivated by the program's existing distinction between persistent relations and changing activity. It is not imposed on all GlassHelix models.

Let `E` be a declared source-to-target relation over typed domains. A member `e` retains its logical identity even when two members have the same source and destination. Let `W` contain trainable shared values on that support. For instance `b`, let `x_b` be its input, `s_b` source-side activity and `a_b` target-side activity. Consider

    q_b = s_b ⊙ x_b
    v_b = W q_b
    y_b = a_b ⊙ v_b

Equivalently, the instance-specific relation is `diag(a_b) W diag(s_b)`. GH may obtain activity from observed context, inferred state, a neural function or a prescribed input. CE need not know whether an activity is intracellular or external. Calling it chromatin activity requires independent evidence.

### Why this is learning structure, not merely a faster generic layer

The model shares a potential influence pattern while allowing its realization to change with context. It can fit common influences across many observations without assigning every cell an independent edge array. The shared structure can serve several observation modalities or regimes when the scientific hypothesis supports it. Its latent activities and influence combinations may remain non-identifiable; shared support is not a proof that those influences are direct molecular regulation.

A fully materialized per-instance edge representation stores `B × |E|` values, potentially more for saved activations. The exact factorization stores shared `|E|` values plus source/target activities of sizes `B × n` and `B × m`. Persistent topology and forward/transpose maps can be prepared once. Whether this wins depends on support density, activity representation, batch size, memory traffic and reuse. If source/target domains are large relative to `|E|`, the activity arrays can outweigh the saving. No universal speedup is asserted.

The cheap first implementation is a **composition**: Torch activity generators and products around a CT-exposed native relation application. A fused CE implementation is optional and must beat this composed reference on complete cost. Do not invent a separate graph planner or assume one layer means one kernel.

### Required derivatives and contribution semantics

For output cotangent `λ_b`, define `u_b = a_b ⊙ λ_b`. Then

    ∂L/∂x_b = s_b ⊙ (Wᵀ u_b)
    ∂L/∂s_b = x_b ⊙ (Wᵀ u_b)
    ∂L/∂a_b = λ_b ⊙ v_b
    ∂L/∂w_e = Σ_b u_b[dst(e)] q_b[src(e)]

Further chain rules pass through any Torch-produced activities. Reuse of a parameter at several model sites or times adds contributions; it does not repeatedly update the value. Duplicate logical edges receive distinct gradients when represented by distinct parameters. A loss reduction's scaling occurs exactly once.

A trainable edge value equal to zero remains in the declared support and can have nonzero gradient. A zero activity can also have a nonzero derivative with respect to that activity. Do not reuse forward zero-skipping as backward support inference. A hard structural exclusion is a different operation. Handle missing measurements through an explicit observation/masking contract, not by treating all measured zeros as missing or all absent counts as inactive regulation.

`references/shared_support_witness.py` checks the factorization, analytic gradients, repeated-use accumulation, zeros and logical permutations against explicit Torch references. It uses CPU double precision and is **not** CE/CT acceptance or biological evidence.

### Scientific rivals and falsification

An unconstrained context-dependent relation cannot generally be represented by two-sided diagonal gates: this is a substantive restriction. Scale transformations can exchange magnitude between `W` and activities; the resulting factor identities are not automatically unique. A fixed support can omit a real changing interaction. A model under this restriction must therefore face a matched baseline and a less-restricted rival where feasible.

Select this candidate only if shared structured influences are scientifically useful at the chosen representation. Arbitrary PCA coordinates do not become genes by attaching labels. Use an anchored gene/module domain where appropriate, or label the construction an effective latent relation and limit its interpretation. Support inferred from training RNA is an inferred modeling assumption; an ATAC-derived support is not permitted in an RNA-only arm.

## Candidate B: differentiable typed multi-input mechanism assembly

Some questions require interactions that should not be flattened into independent pairwise edges. A general form is

    v_bm = φ_m(x_b[i₁], …, x_b[iₖ], θ_m, u_b)
    y_b = declared_assembly_m(v_bm)

Inputs have ordered semantic roles and may repeat the same source. Parameters may be shared by evaluator family or biological context. Stable incidence can be gathered/prepared once and values/activity rebound across observations. The backward scatters and **adds** contributions from repeated arguments and shared parameters through the declared assembly.

NF1A already retains indexed, ordered/repeated incidence and an executable small evaluator vocabulary, but the inspected indexed API does not expose the complete parameter derivative path [S9]. Making one needed evaluator differentiable could be more useful than adding another dense layer. It is also more work than Candidate A. Choose it now only when the initial hypothesis genuinely needs such a multi-input interaction; do not build a universal reaction engine or per-edge interpreted callback system.

Biological restriction and computation must remain separate: the GH model chooses which multi-input process is admissible; CE executes its declared arithmetic and contribution effects. A product or threshold in a formula does not itself prove a biochemical complex.

## Candidate C: observation-aware inference of shared hidden activity

Rather than learning an arbitrary encoder, one can pose hidden-activity reconstruction against typed, incomplete observation channels:

    find r compatible with { observed_k ≈ H_k(C_k r, context_k) }

The scientific object is the compatible activity/forcing set and its role in future evolution. The computational opportunity is repeated application of shared support, observation restrictions and adjoints across many different missingness patterns, with warm starts and reusable preparation. CE could make these repeated structured solves cheaper without owning the scientific uncertainty model.

This can be built initially from Torch optimization plus existing CT/CE applications; a new solver is not automatically justified. It becomes more attractive if repeated hidden-state conditioning is the workload's real bottleneck. It must not impose one shared instantaneous `r` on divergent siblings simply because they share a barcode. The correspondence/latent boundary belongs to the GH hypothesis.

## Other directions to keep available, not implement now

Ancestry-indexed ragged observation operations may preserve multiple sibling measurements without mistaking a clone mean for a cell state. Shared-subexpression execution across **whole competing hypotheses** may save work without averaging incompatible mechanisms. Both align with Cellerator, but must earn their implementation against the actual first workload; neither is a substitute for deciding the learning problem.

## Selection and stopping

Default to A if the evidence/model permits it, because its exact expression and derivative path are compact and use existing CE relation calculus. Use the executable witness and a tiny model immediately; deeper investigation is needed only where a material ambiguity remains. Select B, C or another construction when it gives a better-supported scientific or computational route.

At least one biology-purpose construction should be reasoned through and exercised. A new native implementation is not required when composition already works. Preserve unfavorable results: model rejection, insufficient data, or an optimization slower than Torch each have distinct meanings. An evaluated-but-unpromoted result can be a successful engineering outcome with a correct fallback, not a claim that Cellerator has achieved an advantage.
