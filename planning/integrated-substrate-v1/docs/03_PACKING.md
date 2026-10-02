# Packing: open algorithms, explicit geometry, no compulsory Cellpack

## The seam

Use a cold host interface of the form:

    propose(problem, constraints, objective, caller_workspace) -> candidate realizations
    validate(problem, realization) -> semantic ownership and capability result
    lower(realization, provider, prepare_context) -> prepared native program

These are responsibilities, not a demand for three more global runtimes. Implement on the existing planner/candidate machinery. `seed/cpp/packing_strategy.hpp` and `seed/python/is1/packing.py` are executable interface witnesses; adapt their useful contracts to existing identities rather than installing their namespaces as another public stack.

A problem contains declared operations and argument order/multiplicity; independent logical input/output/state domains; structural and response supports with tagged universes; shapes/precision; required forward/derivative capabilities; output update semantics; immutable structure identity; and optional representative activity/value statistics explicitly labelled as observations. Discovery of a candidate relation and placement of that relation are separate choices. No relation edge may appear or disappear merely to fill a tile.

Constraints include device capabilities, workspace limits, maximum preparation budget, reuse horizon, deterministic requirements, supported capture behavior and whether approximation is authorized. A supplied explicit plan bypasses search but not validity checks. A no-pack route is always available where the operator has an ordinary reference/provider implementation.

## Realization content

Do not freeze the output into feature blocks alone. The useful common part is:

- independent maps for state access, source and destination order, and operation/cohort order;
- exact operation/contribution ownership, ordered argument maps and reduction schedules;
- local layouts, support maps, padding and optional physical replicas with adjoint reduction ownership;
- provider family and a typed/versioned payload; required precision/capabilities;
- cold preparation and migration costs, steady-state traffic/work estimates, provenance and validity guards.

An operation can appear in different forward and response plans. A physical copy does not become a new biological variable. Replicated values gather through a map; gradients accumulate through its transpose, not by arbitrary averaging. Unordered accumulation requires its own numerical policy. A plan cache key includes structure, capabilities, shape/layout, precision, target and relevant statistics policy—not just a hash of coefficient values or a pointer.

## Initial strategies and the comparisons they enable

**Identity/supplied.** No search, explicit caller geometry. Establishes a usable zero-ceremony baseline and protects users whose own domain knowledge beats an optimizer.

**Cellpack adapter.** Reuse existing support extraction, candidate discovery, exact scorer, optimizer, frozen plans, CPK1/CPE2 paths and measured policy evidence. State limitations precisely. Do not reimplement Cellpack in the new layer and call it an alternative.

**Opcode/shared-load cohorts.** Group compatible evaluator/arity/shape jobs, then favor shared input loads. Ordered input slots and private weights remain distinct. Useful when biology-based proximity differs from instruction compatibility. The supplied deterministic reference implements this separately from support-cohort packing.

**Two-sided relation/hypergraph placement.** Optimize source and destination locality independently and retain exact edge ownership. Compare block density, gather traffic and output reduction costs. A greedy supplied-edge tiler is a viable first implementation; a hypergraph method is an alternative, not a dependency of the API. Candidate nomination by sketches is optional; exact checks decide admissibility.

**Mixed region portfolio.** Combine dense islands, sparse residuals, micro-MMA panels and direct vector work under one plan. Dense conversion is allowed only where ownership and amortized cost justify it. A true N=1 operation remains a valid direct path; do not fabricate unused tensor dimensions to claim tensor-core coverage.

**Incremental repair.** Preserve accepted regions and repair affected placements when structure changes. Activity changes can choose an already prepared route without rebuilding the topology. Repacking requires an explicit migration/publication policy and must beat continuing with the old plan over the expected reuse window.

At least identity, a Cellpack adapter, caller-supplied custom strategy and one genuinely different algorithm must execute the same supported problem through the same API. The portfolio is open; adding strategy four must not require modifying strategy one.

## Objective: count the whole job

For a proposed plan p and reuse horizon H, compare a documented estimate such as:

    J(p) = T_prepare(p) + T_migrate(p)
           + H * [T_forward(p) + w_vjp*T_vjp(p) + w_jvp*T_jvp(p)
                  + T_expected_repair(p)]

Memory is a constraint or separate Pareto dimension, not an arbitrary hidden coefficient. The measured use mix supplies the response weights. Estimates may decompose bytes, launches, padding, gather/reduce traffic, live registers, parameter/master traffic and expected activity; do not present a proxy as measured latency. Keep preparation/amortization, quality and supported regimes separate in result records.

If old cost per use is c_old and new is c_new, the simplest repack test is H*(c_old-c_new) > preparation+migration+expected invalidation cost. Hysteresis and a minimum reuse window prevent thrashing; wrong reuse forecasts are measurable outcomes. Automatic selection must respect numerical/derivative equivalence before comparing time. A new restricted matrix model competes as a model choice, not an equivalent faster lowering of an arbitrary dense relation.

## Conformance and adversaries

Test empty work, a single cell, irregular widths, nonidentity row and column orders, repeated arguments, repeated parameters, duplicate output contributions, physical replicas, all-zero primal/nonzero derivative, mismatched universes, stale epochs, capacity exhaustion, unsupported shape/precision, deterministic and declared nondeterministic assembly. Corrupt or incomplete plans fail before partially overwriting output. Ownership coverage checks use logical operation/contribution IDs, not an assumed one-edge-one-job representation.

The supplied reference packers preserve operation identity and ordered arguments while changing state/cohort order. They are semantic test fixtures and proxy-cost examples, not performance-qualified packers. The native task must wrap the actual Cellpack implementation and lower at least one alternative to actual Cellerator execution.
