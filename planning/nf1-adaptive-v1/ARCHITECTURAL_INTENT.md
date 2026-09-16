# Architectural intent: what matters, and what remains delegated

This document is a decision reference, not a mandatory per-edit read. The current sources support extending the existing design; they do not justify another foundation rewrite. Source references are catalogued in SOURCE_REVIEW.md. Requirements carry acceptance, while the following explains the non-obvious choices.

## 1. Keep scientific identity separate from numerical execution

GlassHelix describes quantity roles, scientific mechanism occurrences, supplied dynamics, observations, inference and questions. Cellerator owns numerical operation semantics, preparation, layout, value instances, device resources and response execution. A scientific mechanism can lower to several numerical stages; several stages may later fuse without merging the biological explanations they represent.

Retain the existing GlassHelix core contracts and the S01 assembled_mechanism approach where sound. That partial implementation already owns signature/axis backing memory and adapts compiled callbacks to the common program. It is a host proof of the boundary, not a general evaluator or CUDA implementation. Finish the inspectable arithmetic route in Cellerator; do not fill the gap with a GlassHelix scalar engine. Production must link a real dependency target rather than compile Cellerator source files into a test-only surrogate.

Use a bounded inspectable vocabulary sufficient for the accepted supplied systems, with registered compiled custom blocks for escape hatches. Do not implement a new language, generic JIT, arbitrary automatic differentiation engine or universal graph compiler. Concrete classes and spelling remain delegated. A custom block must declare dependency/effect/capability semantics; a name is not an implementation.

## 2. Extend the prepared program, rather than replace it

The preserved P06 source has typed bindings and owner-specific preflight. It validates every stage before submitting any, then records attempted and accepted submissions. Preserve this distinction: invalid metadata must not partially execute earlier valid stages, but an asynchronous device or callback failure cannot promise rollback of already submitted effects.

Reuse program/session/workspace/provenance owners. Do not infer completion from callback success or counters. A result becomes observable only under the applicable completion/lifetime contract. Add missing ordinary operations where the accepted workload needs them. Avoid a second operation dispatcher hidden in GlassHelix or in a new NF1 namespace.

## 3. Shared structure is not shared mutable state

The V07 source reuses actual prepared-relation projections/maps and allocates independent values/readiness per instance. New siblings inherit no publication; closing the source need not destroy surviving siblings. Retain explicit FP32 authority and the separately requested RNE half projection. Its qualified relation widths are still 1 and 16; arbitrary width is unfinished N work, not already delivered by the value layer.

Definition reuse, support reuse, parameter tying and memoized result reuse are separate operations. Reusing a result requires its full dependency/generation/branch context, not merely equal pointers, stage IDs or evaluator names. Shared computation must retain distinct candidate and entity identities. V07 generations are not historical snapshots, and its external typed FP32 borrowing is explicitly unsupported. A response implementation must own its needed primal data or specify valid deterministic recomputation; do not mistake a generation stamp or a borrowed current pointer for that history.

## 4. N-ary mechanisms and support semantics are real requirements

An ordered incidence structure with repeated arguments and explicit output effects can represent a joint nonlinear mechanism without expanding it into misleading pairwise biological edges. Prefer grouped native evaluation and deterministic accumulation over per-cell Python or per-edge virtual dispatch. Do not materialize cell × edge × argument intermediates by default; a chosen temporary layout must justify its full memory and lifetime cost.

A false predicate excludes evaluation; multiplication by zero is different, notably for nonfinite operands. Primal and derivative support also differ: d(w*x)/dw=x even at w=0. Exact compact-active and persistent-mask realizations must agree under the declared semantics; derivative-relevant terms must not disappear because forward contribution is zero. Both exact realizations are required. Their automatic selection and further fused/specialized realizations are evidence-driven choices.

Approximate dropping remains explicit and assessed: certified, empirical or unassessed. A small coefficient or local derivative is not proof of absent mechanism, especially for another state, input or horizon.

## 5. Differentiate the calculation that was requested

Distinguish the vector field, direct map, implemented RK4 step, observation and implemented rollout. The default response of an RK4 calculation is the derivative of those executed discrete stages, with their exact forcing and branch decisions. A continuous adjoint or reverse ODE is not an interchangeable shortcut. Research on neural ODE adjoints demonstrates why discretization and trajectory reconstruction can change gradients [W2]; this motivates the contract, not adoption of an unrelated framework.

A short rollout may simply retain its required stages. Deterministic checkpoint/recomputation is another admissible implementation. Choose on actual memory/reuse needs, not because a generic adjoint architecture sounds sophisticated. Reverse accumulation must include repeated arguments and fan-out. Changing state, parameters, activity, structure or observation generations invalidates stale requests unless an explicit historical snapshot/recomputation policy supports them. Unsupported or nonsmooth requests fail or report their declared convention; they do not return fabricated zero derivatives.

For a small selected sensitivity matrix, use established SVD or rank-revealing linear algebra directly on the scaled sensitivity. For example, D_y^-1 J D_theta makes the coordinate scaling explicit. Do not form J^T J merely to find the nullspace, and do not hardcode the a+b null direction. Dense small diagnostics are legitimate; they should not force dense Jacobians for all response calculations. Numerical rank remains local, thresholded evidence.

## 6. The observer is not the reduced simulator

An output-irrelevant reporter can reveal a hidden variable through a correlation. Marginalizing it away can destroy the ability to update the inferential state correctly. Keep finite supplied candidate weights and joint correlations separate from physical state values and from computational sharing. Candidate alternatives are not biological population multiplicity, and edge-wise uncertainty must not invent a hybrid mechanism absent from the supplied family.

A supplied reduction is accepted only for its declared preservation/closure obligations. Equal output now does not establish closed reduced evolution. CLUE provides an established example of exact constrained linear lumping for known polynomial ODEs [W3]; it does not justify a universal reduction/discovery engine for NF1. Implement supplied-map residuals and known positive/negative controls. A sampled residual is not a global theorem. Preserve observer/evidence information independently of a physical quotient.

## 7. Optimize the real program, and stop unpromising work

Complete-cost measurement is required. Distinguish preparation, transfer, refresh, support changes, resident stages, response and required output observation. Preserve the proven fallback and compare matched numerical semantics. The original O02–O06 fixed prototype sequence is superseded: test a plausible benefit with the cheapest decisive evidence; build additional specialization only when admissible and useful. Grouped nonlinear execution may already realize the relevant fusion—do not add a redundant optimization subsystem to satisfy a task title.

Use current applicable sparse/vendor/width routes; do not build absent architecture families just to fill a comparison table. Fixed-program CUDA capture and Python device interchange remain optional. They require explicit safety and lifetime evidence when advertised. Negative or not-applicable decisions retain their evidence and reason. No measured speedup is a condition for claiming a correctly delivered baseline toolkit.

## 8. Delegate normal engineering

The binding library, concrete APIs, file naming, exact internal representation, test organization, stage-storage policy and ordinary scope adjustments are not frozen here. The root should choose them from live source and make progress. Preserve a consciously chosen unusual mechanism and its reason; replace incidental scaffolding when a simpler implementation meets the same obligations.

[W2] Gholami, Keutzer and Biros, ANODE, arXiv:1902.10298.
[W3] Ovchinnikov et al., CLUE, arXiv:2004.11961.
Full sources and attribution are in SOURCE_REVIEW.md.
