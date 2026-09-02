# Programming Cellerator

**Status:** Developer-facing guide for the proposed Cellerator 0.1 language
**Research baseline:** Cellerator `main` at `8a56e78a367450d67f6b06bf450279de8379793f`, inspected on 2026-09-01
**Important:** The frontend described here is a language-design proposal. The current repository contains the semantic compiler substrate, not this complete syntax.

**Companion document:** [cellerator-language-specification.md](cellerator-language-specification.md)

## Contents

- [1. The programming model in one page](#1-the-programming-model-in-one-page)
- [2. Enabling Cellerator](#2-enabling-cellerator)
- [3. Domains, axes, state, and relations](#3-domains-axes-state-and-relations)
- [4. Your first relation computation](#4-your-first-relation-computation)
- [5. What happens inside the compiler](#5-what-happens-inside-the-compiler)
- [6. Execution fields](#6-execution-fields)
- [7. Data-aware compilation](#7-data-aware-compilation)
- [8. Branches and bounded profile alternatives](#8-branches-and-bounded-profile-alternatives)
- [9. Persistence and reuse](#9-persistence-and-reuse)
- [10. Hints, preferences, requirements, offers, and force](#10-hints-preferences-requirements-offers-and-force)
- [11. Working with ordinary C++ and CUDA](#11-working-with-ordinary-c-and-cuda)
- [12. Understanding opaque-barrier warnings](#12-understanding-opaque-barrier-warnings)
- [13. Numerical and deterministic programming](#13-numerical-and-deterministic-programming)
- [14. Order, packing, and canonical boundaries](#14-order-packing-and-canonical-boundaries)
- [15. Inspecting what Cellerator did](#15-inspecting-what-cellerator-did)
- [16. IR inspection](#16-ir-inspection)
- [17. Writing an IR transform](#17-writing-an-ir-transform)
- [18. Explicit decomposition](#18-explicit-decomposition)
- [19. Atoms and extents](#19-atoms-and-extents)
- [20. Providing a custom candidate](#20-providing-a-custom-candidate)
- [21. Competing with Cellerator, then forcing it](#21-competing-with-cellerator-then-forcing-it)
- [22. Custom cost and external planning](#22-custom-cost-and-external-planning)
- [23. Asynchronous execution, readiness, and publication](#23-asynchronous-execution-readiness-and-publication)
- [24. Forward, transpose, gradients, and training-shaped programs](#24-forward-transpose-gradients-and-training-shaped-programs)
- [25. The same computation at five control levels](#25-the-same-computation-at-five-control-levels)
- [26. A complete worked example](#26-a-complete-worked-example)
- [27. Common mistakes](#27-common-mistakes)
- [28. Operation-family cookbook](#28-operation-family-cookbook)
- [29. A productive optimization workflow](#29-a-productive-optimization-workflow)
- [30. Building Cellerator libraries](#30-building-cellerator-libraries)
- [31. Compilation and artifact workflow](#31-compilation-and-artifact-workflow)
- [32. Returning to ordinary C++](#32-returning-to-ordinary-c)
- [33. Performance philosophy in practice](#33-performance-philosophy-in-practice)
- [34. Reading the two documents together](#34-reading-the-two-documents-together)
- [35. Source grounding](#35-source-grounding)

## 1. The programming model in one page

Cellerator is C++ that can keep seeing the biology after ordinary C++ would have reduced it to pointers, extents, and loops.

You still control memory. You can still use templates, allocators, pointers, CUDA kernels, intrinsics, libraries, and inline assembly. Cellerator adds a semantic layer for the facts its compiler can exploit:

- what domain an axis belongs to;
- which relation connects which domains;
- which logical edges exist;
- what order the data currently uses;
- which values changed and which structure did not;
- how long a relation, projection, or order will be reused;
- what exact partial results mean;
- which physical candidates are legal;
- where the compiler may plan several operations together.

The smallest useful example is:

```cpp
#pragma cellerator 0.1

#include <cellerator/cellerator.hh>

namespace ce = cellerator;

domain cell;
domain gene;

void propagate(
    state<float, cell, gene> expression,
    relation<float, gene, gene> regulation,
    axis<gene> genes,
    state<float, cell, gene>& response)
{
    <[
        response = expression -[regulation]-> genes;
    ]>
}
```

Read the field body as:

> Apply the gene-to-gene regulatory relation along the gene axis of every cell's expression state, then overwrite `response`. Let Cellerator choose the exact physical realization.

That source does not say CSR. It does not say GEMM. It does not say one kernel. It does not even say GPU. It gives the compiler exact biological semantics and an optimization boundary.

The compiler may select a conventional sparse candidate, preserve a packed order, split by regulatory module, use matrix-engine fragments plus an exact residual, reuse a projection, or fall back to native C++ or CUDA. Every choice must produce the same exact logical result.

This guide starts with that simple layer and then descends, one hatch at a time, into profiles, persistence, effects, decomposition, candidates, IR, and forced physical plans.

## 2. Enabling Cellerator

### 2.1 The pragma

Cellerator syntax is enabled by:

```cpp
#pragma cellerator
```

or, in code that wants a stable language revision:

```cpp
#pragma cellerator 0.1
```

The pragma applies from that line to the end of the current physical file. There is no `end cellerator`.

The mode is file-local. A header that uses Cellerator syntax opts itself in:

```cpp
// regulation.hh
#pragma cellerator 0.1

field void propagate(...) <[
    ...
]>
```

Including that header does not turn the remainder of the including file into Cellerator source. Likewise, a Cellerator-enabled file can include ordinary C++ headers without asking the C++ parser to reinterpret them.

The pragma enables grammar. It does not itself tell the compiler to plan any code. The planning boundary is `<[ ... ]>`.

### 2.2 Ordinary C++ remains ordinary

This is valid Cellerator source:

```cpp
#pragma cellerator 0.1

#include <algorithm>
#include <cstdint>

template<class T>
T clamp_nonnegative(T value)
{
    return std::max<T>(value, T{0});
}
```

No Cellerator planning occurs. The file simply permits Cellerator constructs later.

The deliberate layering is:

```text
#pragma cellerator     the parser understands Cellerator
<[ ... ]>              the planner sees one semantic field
-[relation]->          one typed biological transfer
```

## 3. Domains, axes, state, and relations

### 3.1 Domains are nominal

Declare the biological domains your program manipulates:

```cpp
domain cell;
domain gene;
domain enhancer;
domain regulatory_module;
```

A domain is a type-level identity, not a container. Two domains with the same number of elements are still different.

That lets the compiler reject mistakes that dimensions alone cannot catch:

```cpp
axis<cell> cells;
axis<gene> genes;

state<float, cell> viability;

// Error: this relation expects a gene state, not a cell state.
auto invalid = viability -[regulation]-> genes;
```

Cellerator's standard library can publish familiar domains such as genes, loci, reads, or cells, but you are free to define:

```cpp
domain latent_regulator;
domain trajectory_position;
domain assay_channel;
```

Cellerator is typed biology, not a fixed biological ontology.

### 3.2 Axes carry identity and order

An `axis<gene>` does more than say "this dimension has 20,000 entries." It can carry or resolve:

- a persistent gene-domain identity;
- the current order;
- the extent;
- a partition or hierarchy;
- canonical recovery information;
- geometry identity relevant to planning.

Two gene axes can therefore be statically compatible but runtime-incompatible:

```cpp
axis<gene> reference_genes;
axis<gene> filtered_genes;
```

They have the same domain type, but they may have different persistent identities, extents, or order.

Useful queries are:

```cpp
identityof(genes);
orderof(genes);
extentof(genes);
partitionof(genes);
```

### 3.3 State is a semantic view, not an owner

A declaration such as:

```cpp
state<float, cell, gene> expression;
```

does not allocate memory. It describes a typed view over existing storage.

A standard-library owner might look like:

```cpp
ce::state_buffer<float, cell, gene> expression_storage(
    cell_count, gene_count, ce::device_memory);

state<float, cell, gene> expression =
    expression_storage.view(cells, genes);
```

This separation is important. The compiler reasons about axes, generation, numeric policy, mutability, residency, and readiness. Your allocator and ownership choices remain explicit.

At a lower level, a view can be bound directly:

```cpp
state<float, cell, gene> expression =
    ce::bind_state<float, cell, gene>(
        expression_ptr,
        expression_bytes,
        cells,
        genes,
        expression_generation,
        ce::device(0));
```

### 3.4 Relation structure and relation values have different lives

A weighted relation has two conceptually separate parts:

```cpp
relation_structure<gene, gene> regulatory_topology;
relation_values<float, gene, gene> regulatory_weights;

relation<float, gene, gene> regulation =
    ce::bind_relation(regulatory_topology, regulatory_weights);
```

The structure carries exact logical connectivity, axes, edge identity, logical edge order, and a structure epoch.

The values carry a value generation, numerical representation, readiness, and possibly projection-specific packed forms.

Changing weights does not necessarily invalidate the topology or its projections. Changing the edge set does.

This distinction is one of the main reasons to use Cellerator rather than hand the compiler an anonymous sparse matrix.

### 3.5 Support can change without topology changing

Cellerator distinguishes:

```cpp
support<gene, gene> exact_edges;
active_support<gene, gene> active_edges;
```

`exact_edges` describes logical membership. `active_edges` can be a generation-tagged mask over those same edges.

Turning a stable edge off for one condition changes the active-support generation:

```cpp
ce::update_active_support(active_edges, receptor_predicate);
```

It does not delete the edge from the underlying relation. That lets Cellerator retain structure-only preparation while activity changes.

Actually inserting or deleting edges is structural mutation and advances the structure epoch.

## 4. Your first relation computation

Assume:

```cpp
state<float, cell, gene> expression;
relation<float, gene, gene> regulation;
axis<gene> target_genes;
state<float, cell, gene> response;
```

The ordinary relation transfer is:

```cpp
<[
    response = expression -[regulation]-> target_genes;
]>
```

The assignment means overwrite.

Accumulation is explicit:

```cpp
<[
    response += expression -[regulation]-> target_genes;
]>
```

Affine accumulation is explicit too:

```cpp
<[
    ce::affine(response, alpha, beta) =
        expression -[regulation]-> target_genes;
]>
```

The compiler never gets to guess whether the destination was initialized or whether zeroing it is acceptable.

### 4.1 What the arrow means

For this expression:

```cpp
expression -[regulation]-> target_genes
```

Cellerator checks that:

1. `expression` has a gene axis compatible with the source of `regulation`;
2. `target_genes` is compatible with the destination axis;
3. the relation structure and value generation are current;
4. the result's untouched axes, here `cell`, remain intact;
5. the selected numerical and output contracts are legal.

It then creates a semantic relation-apply operation.

It does not immediately lower that operation to a matrix multiply. The semantic operation remains available to geometry analysis, decomposition, candidate selection, connected planning, and IR inspection.

### 4.2 Selecting an axis explicitly

If a state contains more than one compatible source axis, name the axis:

```cpp
state<float, gene, gene> pair_state;

response =
    pair_state
    -[regulation on regulator_gene_axis]->
    target_genes;
```

This is a semantic disambiguation, not an indexing trick.

### 4.3 Active support

Apply only the currently active edges while preserving the underlying relation identity:

```cpp
<[
    response =
        expression
        -[regulation where active_edges]->
        target_genes;
]>
```

The compiler can now consider candidates that consume a bit mask, a byte mask, a compacted active view, or a fused predicate, provided every one is exact for the declared overlay generation.

### 4.4 Transpose

Transpose is explicit:

```cpp
<[
    source_gradient +=
        destination_gradient
        -[transpose(regulation)]->
        source_genes;
]>
```

Cellerator may use a persistent transpose projection, build one, remap values, or run a direct transpose implementation. Any construction or remapping appears in the cost report.

### 4.5 Chaining relations

A gene-to-module relation followed by a module-to-gene relation can be written:

```cpp
<[
    response =
        expression
        -[gene_to_module]-> modules
        -[module_to_gene]-> genes;
]>
```

The source expresses two biological transformations. The compiler is free to materialize the intermediate, preserve a module-major order, fuse compatible traversal, or use a two-hop candidate.

That freedom only exists because both operations share one field.

## 5. What happens inside the compiler

When the compiler encounters:

```cpp
<[
    response = expression -[regulation]-> genes;
]>
```

the conceptual path is:

```text
parse C++ plus Cellerator syntax
type-check domains and axes
construct semantic operation IR
bind structure, support, values, generation, output, and numeric contracts
attach the representative workload profile
obtain or compile semantic geometry
enumerate exact decompositions and candidate implementations
filter candidates by target, numeric policy, determinism, and memory
price complete preparation and execution
measure or use current evidence when configured
select a correct complete plan
prepare projections, packed operands, and stage bindings
lower selected stages to C++, CUDA, libraries, or backend IR
let the ordinary toolchain optimize and emit native code
```

The current Cellerator source already contains most of these middle layers:

- operation-core and relation-algebra schemas;
- semantic geometry and workload profiles;
- exact decomposition and partial-result algebra;
- candidate catalogs and providers;
- end-to-end and connected-operation planning;
- atom requirements and affordances;
- projection/value-plane separation;
- prepared programs and lowering resumption.

The proposed language gives those layers a coherent source-facing entrance.

## 6. Execution fields

### 6.1 A field is an optimization-visibility boundary

These are two separate planning problems:

```cpp
<[
    module_state =
        expression -[gene_to_module]-> modules;
]>

<[
    response =
        module_state -[module_to_gene]-> genes;
]>
```

This is one connected planning problem:

```cpp
<[
    module_state =
        expression -[gene_to_module]-> modules;

    response =
        module_state -[module_to_gene]-> genes;
]>
```

In the second form, Cellerator can price the transition between operations, preserve a noncanonical intermediate order, fuse stages, or eliminate avoidable packing.

Separate fields can still reuse persistent artifacts. What they cannot do by default is privately erase the visible intermediate or treat both bodies as one graph.

### 6.2 A field does not mean GPU

A field can lower to:

- host C++;
- one or more CUDA kernels;
- cuSPARSE or another native library;
- a hybrid sparse and matrix-engine implementation;
- multiple devices;
- a future accelerator;
- an ordinary manual call surrounded by Cellerator transitions.

The target follows compiler inputs, candidate availability, and constraints.

### 6.3 Fields preserve C++ effects

Inside a field, C++ still matters:

```cpp
<[
    response = expression -[regulation]-> genes;
    log_generation(generationof(response));
    next = response -[regulation]-> genes;
]>
```

If `log_generation` has no effect contract, it is opaque. Cellerator preserves the call and may materialize a boundary before it.

Volatile access, atomics, exceptions, I/O, synchronization, and aliasing similarly constrain reordering.

### 6.4 The planning prologue

Add a planning prologue before `::`:

```cpp
<[
    given ce::uses(regulation) >= 10000;
    prefer ce::minimum_latency;
    require ce::deterministic;
    inspect candidates, costs;
::
    response = expression -[regulation]-> genes;
]>
```

The prologue is declarative. The body is executable.

The categories are intentionally different:

- `given` tells the compiler what you expect;
- `prefer` tells it what you value;
- `require` limits what is legal;
- `offer` contributes an alternative;
- `force` chooses an alternative;
- `inspect` asks what happened.

You will use these progressively through the rest of the guide.

### 6.5 Named fields

Use a named field for reusable semantic computation:

```cpp
field void propagate(
    state<float, cell, gene> expression,
    relation<float, gene, gene> regulation,
    axis<gene> genes,
    state<float, cell, gene>& response)
<[
    response = expression -[regulation]-> genes;
]>
```

Called from ordinary C++, this behaves like a compiled function.

Called inside another field, its semantic body can remain visible:

```cpp
<[
    propagate(expression, regulation, genes, first);
    propagate(first, regulation, genes, second);
]>
```

You may express planning intent:

```cpp
prefer ce::inline_semantics(propagate);
require ce::field_boundary(propagate);
```

Semantic inlining is not the same thing as C++ `inline`. It exposes the field body to Cellerator's planner. Ordinary backend inlining still happens later.

### 6.6 Nested fields

Nested fields are scoped planning overlays:

```cpp
<[
    given ce::persists(structure(regulation), ce::across(experiment));
::
    module_state =
        expression -[gene_to_module]-> modules;

    <[
        prefer ce::minimum_transient_memory;
    ::
        response =
            module_state -[module_to_gene]-> genes;
    ]>
]>
```

The inner operations remain visible to the outer planner. The inner preference applies locally.

To make the inner field a real boundary:

```cpp
<[
    require ce::field_boundary;
::
    response =
        module_state -[module_to_gene]-> genes;
]>
```

Use a boundary when you want separate compilation, stable artifact identity, controlled code size, or deliberate opacity.

## 7. Data-aware compilation

### 7.1 Representative data normally comes from the build

A build might conceptually say:

```text
cellerator compile regulatory.cc
    --profile expression=activated-fibroblast.ceprofile
    --profile regulation=cardiac-regulatory-network.ceprofile
    --target sm70
```

The source does not need to know file paths. It sees symbolic profile bindings.

Profiles may describe degree distributions, support activity, width, recurrence, value dynamics, reuse, memory pressure, or semantic geometry. They never replace exact relation identity.

### 7.2 Refine a profile in source

A field can state an expected data state:

```cpp
<[
    given ce::profileof(expression) matches activated_fibroblast;
::
    response = expression -[regulation]-> genes;
]>
```

This is a planning fact. If the profile is wrong, Cellerator must still produce the correct mathematical result. It may choose a slower plan or fall back after a visible compatibility check.

### 7.3 Let Cellerator propagate state

Recognized semantic operations carry data-state transfer functions.

For example:

```cpp
<[
    normalized =
        ce::segment_normalize(
            expression, tissues, ce::l2);

    response =
        normalized -[regulation]-> genes;
]>
```

The compiler knows that normalization changes value statistics but preserves the expression axes and relation structure.

Likewise:

```cpp
ce::update_active_support(active_edges, receptor_predicate);
```

changes support-overlay generation and activity statistics, not topology.

This automatic state flow is what lets a later operation be planned for the data expected at that point, not merely for the input dataset.

### 7.4 Add an `expect` after custom code

Suppose a custom routine activates fibroblasts:

```cpp
activate_fibroblasts(expression);

expect ce::profileof(expression) matches activated_fibroblast;
```

`expect` helps planning from that point onward. It does not insert a correctness assertion.

Use `verify` when you actually require a check:

```cpp
verify ce::profileof(expression) matches activated_fibroblast;
```

The check may happen at compile time, preparation time, or runtime, depending on what the profile predicate needs.

### 7.5 Missing hints produce warnings, not instant failure

A useful diagnostic might say:

```text
cellerator: profile after call to activate_fibroblasts is unknown
            candidate ranking for relation "regulation" is profile-sensitive
            using generic exact plan
            note: add an effect transfer function or `expect profileof(...)`
```

That is the intended posture. Cellerator can continue, but it tells you where knowledge would help.



## 8. Branches and bounded profile alternatives

Cellerator is built for heavy numerical work, not for compiling an enormous forest of branch-specific executables.

This code is legal:

```cpp
<[
    if (hypoxic) {
        activate_hypoxia_program(expression);
        expect ce::profileof(expression) matches hypoxic_fibroblast;
    } else {
        preserve_baseline(expression);
        expect ce::profileof(expression) matches quiescent_fibroblast;
    }

    response = expression -[regulation]-> genes;
]>
```

At the join, the compiler can:

- form one conservative joined profile;
- retain two bounded alternatives;
- emit two plans plus an explicit selector;
- use a generic exact fallback.

It should not recursively specialize every later branch without a code-size and dispatch policy.

You can make the bounded alternative explicit:

```cpp
<[
    given ce::profileof(expression) in ce::profiles{
        quiescent_fibroblast,
        hypoxic_fibroblast
    };
    require ce::specialization_count <= 2;
::
    response = expression -[regulation]-> genes;
]>
```

When the alternatives do not materially change the plan, one joined executable is usually better. When they do, the optimization report should show the tradeoff:

```text
profile alternative 1: row-masked sparse candidate
profile alternative 2: packed module candidate
dispatch cost: 0.8 us
expected gain over joined plan: 12.4 us
selected: two-way dispatch
```

A runtime profile classifier is itself work. If you offer one, its cost and effects participate in planning.

## 9. Persistence and reuse

### 9.1 Tell the compiler what survives

Packing can be foolish for a one-shot operation and brilliant for a relation reused ten thousand times. Cellerator therefore lets you describe persistence directly.

```cpp
<[
    given ce::persists(
        structure(regulation),
        ce::across(trajectory));

    given ce::changes(
        values(regulation),
        ce::each_iteration);

    given ce::uses(regulation) >= trajectory.size();

    prefer ce::minimum_amortized_latency;
::
    for (auto step : trajectory) {
        response = expression -[regulation]-> genes;
        update_regulatory_values(
            values(regulation), step);
    }
]>
```

This tells the compiler:

- topology is stable across the trajectory;
- edge values change each step;
- structure-only preparation can be amortized;
- value packing may need refresh;
- a dynamic-value candidate may beat a static packed-value candidate.

That is not a cache hint pasted onto biology. It is a description of the biological and computational lifetime.

### 9.2 Different things persist independently

Useful facts include:

```cpp
given ce::persists(structure(regulation), ce::across(experiment));
given ce::persists(orderof(expression), ce::across(experiment));
given ce::persists(projectionof(regulation), ce::across(batch));
given ce::changes(values(regulation), ce::each_iteration);
given ce::support_evolves(active_edges, ce::slowly);
given ce::uses(regulation) == 50000;
```

Do not say that the whole relation is "constant" when only its structure is stable. Cellerator can reuse more when your lifetime description is precise.

### 9.3 Structure epoch versus value generation

The common case:

```cpp
update_regulatory_values(values(regulation), delta);
```

should advance the value generation and preserve the structure epoch.

A structural edit:

```cpp
ce::insert_edges(structure(regulation), new_edges);
```

should advance the structure epoch. That can invalidate:

- logical edge-value alignment;
- active-support overlays;
- exact coverages;
- projections;
- packed operands;
- prepared stages;
- executable artifacts.

The compiler tracks this dependency fan-out automatically.

You can inspect it:

```cpp
inspect persistence, state_flow;
```

A report might say:

```text
regulation.structure
    identity: cardiac.regulation.v3
    epoch: 18 -> 18
    reused: semantic geometry, decomposition, projection

regulation.values
    generation: 742 -> 743
    rebuilt: packed value plane
    reused: structural packing and execution recipe
```

### 9.4 Active support has its own generation

For a stable relation with changing activity:

```cpp
<[
    given ce::persists(structure(regulation), ce::across(experiment));
    given ce::support_evolves(active_edges, ce::every(batch));
::
    ce::update_active_support(active_edges, receptor_predicate);

    response =
        expression
        -[regulation where active_edges]->
        genes;
]>
```

Only the overlay generation changes. A candidate can consume the mask directly, compact it, or use it as a fused gate. The selected plan must still account for the update and any compaction cost.

### 9.5 Persistent order is a performance asset

Consider two connected relations:

```cpp
<[
    modules =
        expression -[gene_to_module]-> module_axis;

    response =
        modules -[module_to_gene]-> gene_axis;
]>
```

The first operation may naturally produce module state in an order that the second operation already accepts. Cellerator can preserve that order and avoid a gather.

This is why you should not canonicalize every intermediate by habit:

```cpp
modules = ce::canonicalize(
    expression -[gene_to_module]-> module_axis);
```

Use canonicalization when an external observer requires it, not as ritual hand-washing between operations.

### 9.6 Biological epoch boundaries

A long-running program may have periods during which topology is stable:

```cpp
for (auto developmental_stage : stages) {
    ce::begin_epoch(regulation, developmental_stage);

    <[
        given ce::persists(
            structure(regulation),
            ce::across(developmental_stage));
    ::
        run_stage(regulation, expression);
    ]>

    ce::end_epoch(regulation, developmental_stage);
}
```

The standard-library epoch objects can be biological, experimental, or purely computational. Cellerator cares about the lifetime guarantee, not the label's ontology.

Normal owners should advance generations and epochs automatically. Experts can state and manipulate them directly when integrating custom storage or persistence.

## 10. Hints, preferences, requirements, offers, and force

These five levels are intentionally not synonyms.

### 10.1 `given`: a fact for economics

```cpp
given ce::uses(regulation) >= 10000;
given ce::profileof(expression) matches activated_fibroblast;
```

A `given` fact can change candidate ranking. It cannot make an incorrect candidate legal.

Use it for expected workload and data facts.

### 10.2 `prefer`: a soft objective

```cpp
prefer ce::minimum_latency;
prefer ce::minimum_transient_memory;
prefer ce::avoid_transfer;
```

A preference can lose when another cost dominates.

You can combine weighted preferences:

```cpp
prefer 4 * ce::minimum_latency
     + 1 * ce::minimum_persistent_memory;
```

The optimization report should show the resulting frontier rather than pretending one scalar objective fell from the sky.

### 10.3 `require`: a hard boundary

```cpp
require ce::deterministic;
require ce::canonical_output(response);
require ce::persistent_bytes <= 2_GiB;
require ce::no_host_synchronization;
```

If no complete legal plan satisfies the requirements, compilation fails.

Use `require` when violating the condition would make the program unacceptable, not merely slower.

### 10.4 `offer`: compete with the compiler

```cpp
offer decomposition by_module;
offer candidate my_sparse_candidate;
offer transform paired_moment_pass;
offer cost_model cluster_costs;
```

The compiler validates every offer and then lets it compete.

This is the power-user sweet spot. You contribute expertise without disabling the rest of the compiler.

### 10.5 `force`: choose deliberately

```cpp
force decomposition by_module;
force candidate my_sparse_candidate;
```

Use force for controlled experiments, production lock-down, or cases where you know something the planner cannot represent.

Force does not mean "skip validation." It means "among validated meanings, use this one."

A complete physical plan can be fixed with:

```cpp
force realization sm70_locked_plan;
```

If it cannot run on the target, compilation fails unless you supplied an explicit fallback:

```cpp
force ce::first_available{
    sm70_locked_plan,
    conventional_sparse
};
```

### 10.6 The full hierarchy in one field

```cpp
<[
    given ce::persists(
        structure(regulation),
        ce::across(trajectory));

    prefer ce::minimum_amortized_latency;

    require ce::deterministic;
    require ce::persistent_bytes <= 2_GiB;

    offer decomposition by_regulatory_module;
    offer candidate lab_hybrid_candidate;
    offer cost_model measured_machine_room_cost;

    inspect candidates, costs, persistence;
::
    response = expression -[regulation]-> genes;
]>
```

The order in the prologue is mainly for reading. The semantic authority is:

```text
correctness
require
validated force
offer
prefer
given
compiler defaults
```

## 11. Working with ordinary C++ and CUDA

### 11.1 Unknown calls are honest barriers

This field contains a call Cellerator does not understand:

```cpp
<[
    modules =
        expression -[gene_to_module]-> module_axis;

    hand_tuned_module_kernel(modules);

    response =
        modules -[module_to_gene]-> gene_axis;
]>
```

Cellerator must assume the call can observe or mutate anything reachable. It may have to:

- materialize `modules`;
- establish a concrete order;
- wait for readiness;
- abandon profile propagation;
- restart planning after the call.

That is conservative, but it is safe and humble.

### 11.2 Add an effect contract

Suppose the custom function changes values but preserves axes and order:

```cpp
void hand_tuned_module_kernel(
    state<float, cell, regulatory_module>& modules)
effects(
    mutates(modules),
    advances(generationof(modules)),
    preserves(
        identityof(modules),
        orderof(modules)),
    deterministic
);
```

Now Cellerator can reason through the call.

A richer relation update might say:

```cpp
void update_regulatory_values(
    relation_values<float, gene, gene>& weights,
    const float* delta)
effects(
    reads(delta),
    mutates(weights),
    advances(generationof(weights)),
    publishes(weights),
    preserves(
        structure(weights),
        supportof(weights),
        orderof(weights)),
    deterministic
);
```

This lets the compiler retain structure and projection work while rebinding the new value generation.

### 11.3 Effects are not ignorable decoration

C++ attributes are often designed so unknown attributes can be ignored. Cellerator effect contracts alter semantic analysis and cannot be silently discarded.

A visible function body should be checked against its contract where possible. A separately compiled function exports the contract as part of its interface.

A lie in an effect contract is not a clever optimization. It is an invalid program.

### 11.4 Useful effect terms

The main terms are:

```cpp
effects(
    reads(x),
    writes(y),
    mutates(z),
    preserves(structure(r), orderof(r)),
    invalidates(profileof(x)),
    advances(generationof(y)),
    publishes(y),
    reorders(y),
    transfers(y),
    allocates,
    synchronizes,
    deterministic
);
```

Use `effects(opaque)` when a barrier is intentional and you do not want a missing-contract warning.

Use `pure` only when the callable truly has no externally visible effect except its result and declared reads.

### 11.5 Manual CUDA remains CUDA

You can always write:

```cpp
manual_regulation_kernel<<<grid, block, shared, stream>>>(
    expression_ptr,
    row_offsets,
    source_indices,
    weights,
    response_ptr);
```

Cellerator does not reinterpret the launch geometry.

Inside a field, give the wrapper an effect contract or publish it as a candidate. Outside a field, it is ordinary CUDA.

This is the final escape hatch: the compiler is allowed to be ambitious, never imperial.

## 12. Understanding opaque-barrier warnings

A useful warning should tell you what knowledge was lost:

```text
cellerator: opaque call `hand_tuned_module_kernel`
            prevents joint planning across operations 4 and 6

lost facts:
    module_state order
    module_state profile
    value generation transition
    synchronization behavior

inserted boundary:
    materialize persistent module order
    await generation 81
```

You have several legitimate responses:

1. leave the barrier alone;
2. add an effect contract;
3. move the call outside the field;
4. wrap the call as a custom candidate;
5. put the call in a named field with visible semantic IR;
6. force a boundary intentionally.

Cellerator should never pressure you into annotating everything. It should explain the cost of opacity and let you decide.

## 13. Numerical and deterministic programming

### 13.1 State the numerical contract, not a favorite kernel

```cpp
<[
    require ce::numeric(response) == ce::numeric_policy{
        .relation_storage = ce::f16,
        .state_storage = ce::f16,
        .multiply = ce::f16,
        .accumulate = ce::f32,
        .output_storage = ce::f32,
        .rounding = ce::nearest_even,
        .nan = ce::propagate,
        .infinity = ce::propagate
    };
::
    response = expression -[regulation]-> genes;
]>
```

This permits any physical implementation that satisfies the contract.

You have not asked for tensor cores. You have asked for a numerical meaning that might make them legal.

### 13.2 Require determinism when you need it

```cpp
require ce::deterministic;
require ce::fixed_reduction_tree;
```

That can reject candidates using nondeterministic atomics or an unstable partial merge.

A softer request is:

```cpp
prefer ce::deterministic_when_practical;
```

The report should say whether the winner is bitwise deterministic, structurally deterministic, or merely reproducible within a documented tolerance.

### 13.3 Empty segments and nonfinite values

Do not let backend defaults define biology.

For a segmented maximum, softmax, or normalization, choose a standard operation whose contract states:

- the empty result;
- singleton behavior;
- NaN handling;
- infinity handling;
- backward behavior.

The compiler can then compare candidates without changing edge cases.

### 13.4 Aliasing and in-place work

Ask explicitly for in-place execution:

```cpp
require ce::in_place(expression);

expression =
    expression -[regulation]-> genes;
```

This compiles only if the semantic operation and a candidate both support the alias.

Without that contract, pointer equality does not grant permission.



## 14. Order, packing, and canonical boundaries

### 14.1 Logical order and physical order are different

A relation has a stable logical edge order. A candidate may use a different physical order for execution.

Similarly, a state may be carried internally in a persistent order that is better for connected operations.

Cellerator must preserve the maps between these orders. Equal extents are not enough.

### 14.2 Packed output is permission, not a new meaning

You can let a field return a noncanonical order:

```cpp
<[
    prefer ce::packed_output(module_state);
::
    module_state =
        expression -[gene_to_module]-> modules;
]>
```

A later Cellerator operation can consume that order if compatible.

Ordinary code that expects canonical order should request it:

```cpp
canonical_modules =
    ce::canonicalize(module_state);
```

Canonicalization has a measurable cost. It is visible in the plan.

### 14.3 Make external boundaries explicit

A foreign library may require canonical contiguous memory:

```cpp
<[
    module_state =
        expression -[gene_to_module]-> modules;

    canonical_modules =
        ce::canonicalize(module_state);
]>

foreign_library(canonical_modules.data());
```

This cleanly ends Cellerator's freedom at the boundary.

Alternatively, write an adapter that accepts the current order and exposes an effect contract.

### 14.4 Why this matters in chains

The current relation-chain machinery distinguishes a persistent-order path from a materialized recovery-map path. In source terms:

```cpp
<[
    module_state =
        expression -[gene_to_module]-> modules;

    response =
        module_state -[module_to_gene]-> genes;
]>
```

can avoid an intermediate reorder if both operations agree on the module order.

The plan report should show:

```text
boundary gene_to_module -> module_to_gene
    producer order: module.pack.17
    consumer accepts: module.pack.17
    transform: none
    saved bytes: 128 MiB
```

or, when they disagree:

```text
transform: explicit gather
bytes: 128 MiB
cost: 41.2 us
```

## 15. Inspecting what Cellerator did

### 15.1 Start with `inspect`

Add inspection requests to a field:

```cpp
<[
    inspect semantics, state_flow;
    inspect candidates, costs;
    inspect persistence, barriers;
::
    response = expression -[regulation]-> genes;
]>
```

The compiler can also expose equivalent command-line switches. Source requests are useful in examples, regression tests, and libraries whose performance assumptions deserve durable evidence.

### 15.2 Semantic report

A semantic report should read more like a compiler explanation than a profiler dump:

```text
field cardiac::propagate

operation 0
    kind: relation_apply
    source state: expression
    source axis: cardiac::gene / order canonical-gene-v4
    relation: cardiac.regulation.v3
    structure epoch: 18
    value generation: dynamic
    destination axis: target_genes
    output: response
    update: overwrite
    output order: packed permitted
    numeric: f16 x f16 -> f32
```

This lets you check that the compiler understood the biology before asking whether it optimized well.

### 15.3 State-flow report

```text
program point 3
    expression generation: 41
    expression profile: activated-fibroblast / confidence 0.91
    regulation structure epoch: 18
    regulation value generation: 742
    active support generation: 9

after update_regulatory_values
    preserved: structure, axes, edge order, projection
    advanced: value generation 742 -> 743
    invalidated: packed value plane
```

The report should separate exact facts, inferred facts, given facts, and unknowns.

### 15.4 Candidate report

```text
candidate row-masked-n1
    legal
    analytical total: 112.7 us

candidate v100-dense-fragment-plus-residual
    legal
    measured median: 78.4 us
    projection build: 6.2 ms
    amortized at reuse 10,000: 79.0 us
    selected

candidate cusparse-csr
    rejected: graph capture required
```

A useful report includes every serious alternative, not just the winner.

### 15.5 Cost report

Cellerator should report complete cost:

```text
host preparation              12.0 us
semantic packing               0.0 us reused
projection construction        6.2 ms reused across 10,000
static value packing           0.0 us
dynamic value packing          8.1 us
host-to-device transfer        0.0 us
kernel                         68.9 us
epilogue                        1.1 us
order transform                 0.0 us
synchronization                 0.9 us
communication                   0.0 us
amortized total                79.6 us
```

Kernel time alone can be a mirage with expensive shoes.

### 15.6 Counterfactual questions

A mature compiler should let you ask:

```cpp
inspect ce::compare(
    ce::current_plan,
    ce::without(
        ce::persists(structure(regulation))),
    ce::with(
        ce::uses(regulation) == 1));
```

The answer might show that a packed projection wins only beyond 320 reuses.

Counterfactual inspection does not change the program. It helps you decide whether your persistence model is worth expressing.

## 16. IR inspection

### 16.1 The IR ladder

The proposed public levels are:

```text
semantic
geometry
decomposition
cover
projection
packed
executable
native
```

You can dump one:

```cpp
inspect ir<semantic>;
inspect ir<projection>;
inspect ir<native>;
```

Each level answers a different question.

`semantic` asks: what biological computation did the compiler understand?

`geometry` asks: what target-independent structure and workload evidence did it derive?

`decomposition` asks: how may the exact work be split?

`cover` asks: which exact pieces own which contributions?

`projection` asks: which target-specific views are used?

`packed` asks: what data must be laid out or refreshed?

`executable` asks: what stages, dependencies, candidates, and workspaces will run?

`native` asks: what reached C++, LLVM, PTX, or another backend?

### 16.2 Bind IR as a compile-time object

For a named field:

```cpp
consteval ir<semantic> graph =
    ir_of<semantic>(propagate);
```

The returned object is a typed immutable view. You can write compile-time assertions:

```cpp
static_assert(
    ce::ir::count<ce::relation_apply>(graph) == 1);

static_assert(
    ce::ir::preserves_domain<cell>(graph));
```

This makes compiler understanding testable.

### 16.3 Inspect profile-specific IR

The same field may lower differently for different profiles:

```cpp
consteval auto activated_plan =
    ir_of<decomposition>(
        propagate,
        activated_fibroblast);

consteval auto quiescent_plan =
    ir_of<decomposition>(
        propagate,
        quiescent_fibroblast);
```

Semantic IR should remain equivalent. Lower IR may differ.

### 16.4 Semantic inlining

Expose a named field's body to a caller:

```cpp
<[
    require ce::inline_semantics(propagate);
::
    propagate(expression, regulation, genes, first);
    propagate(first, regulation, genes, second);
]>
```

Cellerator can now plan the two calls as one graph.

After lowering, Clang or GCC may still inline the generated native functions. These are separate decisions at separate levels.

### 16.5 Human dumps are not the programming API

A textual IR dump is for reading:

```text
%response = ce.relation.apply
    %expression, %regulation
    source_axis = gene.canonical.v4
    destination_axis = gene.canonical.v4
    update = overwrite
```

Do not build tools by parsing that text. Use the typed `ir<level>` API.

The dump may change for readability. The typed schema carries the version contract.

## 17. Writing an IR transform

Suppose two edge statistics traverse the same relation separately. You want to offer a paired traversal.

A transform can add that alternative:

```cpp
transform ir<semantic> pair_regulatory_moments(
    ir<semantic> graph)
{
    auto edit = ce::ir::rewrite(graph);

    edit.match(
            ce::ir::two_moments_of_same_relation())
        .replace_with(
            ce::ir::paired_relation_moments());

    return edit.commit();
}
```

Offer it:

```cpp
field void analyze_regulation(...) <[
    offer transform pair_regulatory_moments;
    inspect ir<semantic>, candidates, costs;
::
    first_moment =
        ce::relation_moment(
            expression, regulation, ce::first);

    second_moment =
        ce::relation_moment(
            expression, regulation, ce::second);
]>
```

The transform does not directly emit a kernel. It creates a verified semantic alternative that can have several physical candidates.

### 17.1 Transform rules

A transform must:

- preserve or explicitly map stable identities;
- preserve exact outputs and effects;
- declare any new numeric assumptions;
- pass the destination-level verifier;
- expose its provenance in diagnostics;
- use caller-owned or compiler-managed compile-time storage rather than hidden runtime allocation.

If it cannot prove semantic equivalence, publish a new semantic operation or candidate instead.

### 17.2 Intercept a lower stage

An expert projection builder might be offered as:

```cpp
offer ce::lowering<projection>(
    my_module_projection);
```

A packed-value builder might be forced:

```cpp
force ce::lowering<packed>(
    my_generation_aware_packer);
```

Lowering hooks receive typed IR and explicit capacities. They return validated output and costs.

They do not get a trapdoor around exact coverage or generation checks.

## 18. Explicit decomposition

### 18.1 Start with the biological split

Suppose the regulatory relation contains stable modules that can be processed independently.

Define exact coverages:

```cpp
coverage module_coverages =
    ce::cover_by(
        regulation,
        regulatory_modules);
```

Define how partial responses combine:

```cpp
partial_algebra response_sum =
    ce::partial_algebra<float>(
            "cardiac.response.sum.v1")
        .neutral(0.0f)
        .merge(ce::add)
        .finalize(ce::identity)
        .associative()
        .commutative()
        .numeric(ce::f32_accumulation);
```

Build a decomposition:

```cpp
decomposition by_module =
    ce::decompose(regulation)
        .split(
            ce::semantic_components(
                module_coverages))
        .output(
            ce::partial(
                response,
                response_sum))
        .fallback(ce::unsplit);
```

Offer it:

```cpp
<[
    offer decomposition by_module;
    inspect ir<decomposition>, candidates, costs;
::
    response =
        expression -[regulation]-> genes;
]>
```

### 18.2 Exact coverage is not clustering

A support cluster, affinity group, or sampled module can propose a split. It becomes executable only after exact certification.

The decomposition must identify every logical edge contribution exactly. It must also say whether coverages overlap and who owns the result.

### 18.3 Halos and replicas

A destination split may need source values from neighboring coverages:

```cpp
decomposition by_destination =
    ce::decompose(regulation)
        .split(ce::destination_axis(partitions))
        .read_halo(source_halo)
        .replicate(shared_regulators)
        .output(ce::exclusive(destination_owners))
        .fallback(ce::unsplit);
```

A halo is read-only. A replica is physical. Neither becomes an output contributor merely because it is present.

### 18.4 Ordered partials

Not every merge is commutative.

```cpp
partial_algebra ordered_scan =
    ce::partial_algebra<scan_state>(
            "ordered.scan.v1")
        .neutral(initial_scan_state)
        .merge(scan_merge)
        .finalize(scan_finalize)
        .ordered_only(trajectory_order)
        .deterministic_tree(scan_tree);
```

Cellerator can only reorder or parallelize within that algebra's exact rules.

### 18.5 Split dimensions

The current compiler architecture supports meaningful split families such as:

```cpp
ce::source_axis(...)
ce::destination_axis(...)
ce::relation_edges(...)
ce::semantic_components(...)
ce::segments(...)
ce::modules(...)
ce::extents(...)
```

Choose a split because it corresponds to exact reuse, locality, ownership, or algebra. Do not choose one because its name sounds biological.

### 18.6 Let your decomposition lose

An offered decomposition may be correct and still bad.

Perhaps its halos are large. Perhaps module imbalance dominates. Perhaps assembly cost erases the kernel gain.

That is why `offer` exists. Your structure enters the tournament without crowning itself.

## 19. Atoms and extents

### 19.1 An atom is relative to a compiler level

An atom is not "the smallest piece of biology."

At one level an atom may be an exact semantic coverage. At another it may be a physical structure plane plus mutable values. At another it may be a prepared executable fragment.

What matters is that it is independently nameable, bindable, invalidatable, and composable at that level.

### 19.2 Atom planes

An expert atom may expose separate planes:

```text
immutable structural plane
mutable relation-value plane
active-support overlay
mutable state plane
gradient plane
partial-result plane
dense-result plane
physical projection
readiness and lease state
```

This mirrors the real lifetime split. Stable topology can survive many value and activity generations.

### 19.3 Candidate requirements

A candidate can request:

```cpp
atom_requirement relation_piece =
    ce::atom_requirement()
        .coverage(module_coverage)
        .plane(ce::structural_plane)
        .plane(ce::relation_values_plane)
        .order(module_major_order)
        .index_width(ce::u32)
        .alignment(128)
        .contiguity(ce::multi_extent_allowed)
        .generation(ce::current)
        .graph_stable_address();
```

An available atom advertises affordances. Binding is a typed compatibility check, not pointer matching.

### 19.4 Multi-extent inputs

A logical port may be backed by several extents:

```cpp
extent left = ce::bind_extent(...);
extent right = ce::bind_extent(...);

auto binding =
    ce::bind_port(
        module_input,
        ce::extents{left, right});
```

A candidate may consume them directly. Another candidate may require a contiguous assembly.

The planner prices both:

```text
candidate direct-multi-extent
    launch: 2
    assembly: 0
    kernel: 91 us

candidate assembled-dense
    assembly: 24 us
    kernel: 52 us
    total: 76 us
```

No hidden copy sneaks under the floorboards.

### 19.5 Local index width

A huge relation can consist of many independently bounded components:

```cpp
auto hierarchy =
    ce::hierarchical_index_space(
        global_extent,
        ce::components{
            ce::component(id0, local_map0, ce::u16),
            ce::component(id1, local_map1, ce::u32)
        });
```

The compiler can use compact local indices without pretending the global relation fits in 16 or 32 bits.

Global recovery remains explicit.



## 20. Providing a custom candidate

### 20.1 Wrap an implementation, do not hide it

Assume you wrote a CUDA implementation optimized for a recurring regulatory support pattern.

Build a candidate descriptor:

```cpp
candidate lab_hybrid_candidate =
    ce::candidate(
            "alexanian.regulation.sm70.hybrid.v1")
        .supports(ce::relation_apply)
        .requires(
            ce::target_capability("nvidia.sm70"))
        .accepts(
            ce::numeric_policy{
                .relation_storage = ce::f16,
                .state_storage = ce::f16,
                .multiply = ce::f16,
                .accumulate = ce::f32,
                .output_storage = ce::f32
            })
        .projection(lab_projection_abi)
        .atoms(lab_atom_requirements)
        .resources(query_lab_resources)
        .prepare(prepare_lab_candidate)
        .launch(launch_lab_candidate)
        .effects(lab_candidate_effects)
        .cost(lab_analytical_cost)
        .measure(measure_lab_candidate);
```

The implementation functions are ordinary C++ or CUDA.

The candidate object tells Cellerator:

- which semantic operations it realizes;
- what exact data and physical views it needs;
- what target and numerical contracts it supports;
- how much workspace it needs;
- how to prepare and launch it;
- how to price and measure it.

### 20.2 Offer it locally

```cpp
<[
    offer candidate lab_hybrid_candidate;
    inspect candidates, costs;
::
    response =
        expression -[regulation]-> genes;
]>
```

The candidate competes only in that field.

A library can publish source-linked candidate fragments so every importing field can discover them during compilation or preparation.

### 20.3 Why a candidate can be rejected

A correct kernel can still be illegal for a particular operation.

Common reasons include:

- destination order mismatch;
- no exact logical recovery map;
- missing transpose closure;
- unsupported value generation mode;
- wrong accumulation type;
- nondeterministic atomics under a determinism requirement;
- graph-capture incompatibility;
- insufficient alignment;
- too many extents;
- missing target capability;
- persistent or transient memory limit;
- unsupported active-support representation.

The rejection report should identify the first exact contract that failed and any repair route.

### 20.4 Measurement is part of candidate quality

Analytical cost is useful for filtering and shortlisting. It is not always decisive.

A measurement hook should publish:

- sample count;
- median or another declared statistic;
- spread;
- contamination;
- profile and target identity;
- build and evidence revision;
- complete phase costs where measurable.

Cellerator can then select empirically or fall back analytically when measurement is unavailable.

### 20.5 One provider, several candidates

A source-linked provider can publish:

- a conventional sparse candidate;
- a small-width candidate;
- a transpose candidate;
- a matrix-engine fragment candidate;
- an active-support candidate;
- an exact residual;
- a multi-extent candidate.

Do not make one giant launch function switch internally among them. Separate candidates give the planner visible alternatives and honest costs.

## 21. Competing with Cellerator, then forcing it

### 21.1 Start by offering

```cpp
<[
    offer decomposition by_module;
    offer candidate lab_hybrid_candidate;
    prefer ce::minimum_latency;
    inspect candidates, costs;
::
    response =
        expression -[regulation]-> genes;
]>
```

Suppose the report says:

```text
winner: built-in row-masked candidate
reason: offered hybrid saves 31 us kernel time
        but adds 44 us dynamic value packing
```

That is useful information. Perhaps your candidate needs a projection-primary value owner or a longer reuse horizon.

### 21.2 Change the facts, not the verdict

If the real workload reuses packed values, describe that:

```cpp
given ce::persists(
    packed_values(regulation),
    ce::across(64_steps));
```

Then rerun the plan. Do not force the candidate merely to make the benchmark flattering.

### 21.3 Force a candidate for an experiment

```cpp
force candidate lab_hybrid_candidate;
```

Now the compiler selects that candidate wherever its force scope applies, while still choosing compatible decomposition, projection, and binding details.

Use this for:

- controlled comparisons;
- diagnosing the planner;
- reproducing a result;
- locking a known production choice;
- exploiting knowledge not yet expressible in the cost model.

### 21.4 Force a decomposition but not a kernel

```cpp
force decomposition by_module;
```

Cellerator must use the module split, but it can still choose the best candidate for each exact fragment and the best partial merge.

This is often more portable than forcing one target-specific candidate.

### 21.5 Force a complete realization

```cpp
force realization sm70_locked_plan;
```

This can lock:

- exact decomposition;
- coverage and ownership;
- projections;
- packed operands;
- candidate assignments;
- stage dependencies;
- resource requirements.

It remains validated against current structure epoch, generations, target, toolchain, and numerical contract.

### 21.6 Drop fully to manual code

When you truly want total control:

```cpp
void manual_propagate(...)
{
    build_or_reuse_my_layout(...);

    my_kernel<<<grid, block, shared, stream>>>(
        ...);
}
```

That is not failed Cellerator. It is ordinary systems programming.

You can later bring it back into the compiler as an effect-contracted call or candidate.

## 22. Custom cost and external planning

Cellerator's local planner understands preparation, packing, transfer, kernel, order, synchronization, memory, and reuse.

A larger system may know more:

- storage fetch cost;
- cross-node transfer;
- queue pressure;
- GPU availability;
- artifact residency;
- expected reuse across jobs.

Offer an external cost model:

```cpp
<[
    offer cost_model machine_room_cost;
::
    response =
        expression -[regulation]-> genes;
]>
```

The model can reprice candidates:

```cpp
cost_model machine_room_cost =
    ce::external_cost()
        .persistent_byte_ns(...)
        .transient_byte_ns(...)
        .transfer_byte_ns(...)
        .communication_byte_ns(...)
        .launch_ns(...)
        .synchronization_ns(...)
        .expected_reuse(...);
```

The compiler can exchange a bounded frontier with the external planner. The external model may say which candidate is globally cheaper.

It cannot waive local correctness. Storage location and device ordinal never replace biological identity.

## 23. Asynchronous execution, readiness, and publication

### 23.1 Field exit is not a host wait

A field can enqueue work:

```cpp
<[
    response =
        expression -[regulation]-> genes;
]>
```

Exiting the field need not synchronize the host. `response` carries a promised generation plus readiness state.

A later compatible Cellerator operation can depend on it:

```cpp
<[
    first =
        expression -[regulation]-> genes;

    second =
        first -[regulation]-> genes;
]>
```

The planner builds stage dependencies and stream ordering.

### 23.2 Await before host observation

Before ordinary host code reads device-produced data:

```cpp
ce::await(response);
consume_on_host(response);
```

The synchronization is explicit and appears in cost diagnostics.

### 23.3 Publish only successful generations

A candidate that promises generation 43 must not publish it if enqueue fails.

Cross-stream consumers validate that:

- the ready event belongs to the expected provider;
- the generation matches;
- the event has not failed;
- any required lease permits access.

This may sound fussy, but stale biological state moving at GPU speed is merely a faster wrong answer.

### 23.4 Rebinding pointers

A prepared executable may allow new pointers and streams without rebuilding its immutable graph:

```cpp
auto prepared = ce::prepare(propagate, profile, target);

prepared.launch(
    ce::bind(expression_next),
    ce::bind(response_next),
    stream_next);
```

Structure epoch, order, value generation, capacity, and candidate requirements are still validated.

## 24. Forward, transpose, gradients, and training-shaped programs

Cellerator can express training-oriented primitives without owning a model, loss, optimizer, or framework.

A field may contain:

```cpp
<[
    forward =
        source -[regulation]-> destination_genes;

    source_gradient +=
        destination_gradient
        -[transpose(regulation)]->
        source_genes;

    value_gradient =
        ce::contract_on(
            supportof(regulation),
            source,
            destination_gradient,
            ce::multiply);

    ce::sparse_update(
        values(regulation),
        selected_edges,
        value_delta,
        ce::add);
]>
```

The compiler can choose a family of forward, transpose, edge-gradient, and update candidates that share structure and projections.

Require graph capture when the workload needs it:

```cpp
require ce::graph_capture_compatible;
```

A captured program separates immutable stages from mutable launch bindings and update policy. Changing pointers or values does not necessarily reconstruct the graph.

Cellerator keeps the low-level execution semantics. GlassHelix or another model layer supplies model meaning.

## 25. The same computation at five control levels

This section shows progressive disclosure for one regulatory propagation.

### Level 1: Cellerator proper

```cpp
<[
    response =
        expression -[regulation]-> genes;
]>
```

You supply semantics. Cellerator chooses everything else.

### Level 2: planning knowledge

```cpp
<[
    given ce::persists(
        structure(regulation),
        ce::across(trajectory));

    given ce::uses(regulation) >= 10000;

    prefer ce::minimum_amortized_latency;
::
    response =
        expression -[regulation]-> genes;
]>
```

You describe workload economics.

### Level 3: hard contract

```cpp
<[
    require ce::deterministic;
    require ce::canonical_output(response);
    require ce::persistent_bytes <= 2_GiB;
::
    response =
        expression -[regulation]-> genes;
]>
```

You constrain the legal solution space.

### Level 4: offer compiler structures

```cpp
<[
    offer decomposition by_regulatory_module;
    offer candidate lab_hybrid_candidate;
    offer transform paired_moment_pass;
    inspect candidates, costs, ir<executable>;
::
    response =
        expression -[regulation]-> genes;
]>
```

Your ideas compete with built-ins.

### Level 5: fix the physical realization

```cpp
<[
    force realization sm70_locked_plan;
    inspect ir<native>;
::
    response =
        expression -[regulation]-> genes;
]>
```

You select the complete validated plan.

Beyond Level 5 lies ordinary manual C++ or CUDA.

The language never changes species as you descend. It simply reveals more of the compiler's skeleton.

## 26. A complete worked example

The following example combines the ordinary and expert layers without requiring every reader to use the expert pieces.

```cpp
#pragma cellerator 0.1

#include <cellerator/cellerator.hh>
#include <cellerator/bio/cardiac.hh>

namespace ce = cellerator;

namespace cardiac {

domain cell;
domain gene;
domain regulatory_module;

void update_regulatory_values(
    relation_values<float, gene, gene>& weights,
    const state<float, cell, gene> expression,
    ce::step_id step)
effects(
    reads(expression, step),
    mutates(weights),
    advances(generationof(weights)),
    publishes(weights),
    preserves(
        structure(weights),
        supportof(weights),
        orderof(weights)),
    deterministic
);

partial_algebra module_sum =
    ce::partial_algebra<float>(
            "cardiac.module-sum.v1")
        .neutral(0.0f)
        .merge(ce::add)
        .finalize(ce::identity)
        .associative()
        .commutative()
        .numeric(ce::f32_accumulation);

decomposition by_module =
    ce::decompose(ce::operation::relation_apply)
        .split(ce::semantic_components(
            ce::bio::regulatory_modules))
        .partials(module_sum)
        .fallback(ce::unsplit);

field void simulate_activation(
    state<float, cell, gene>& expression,
    relation<float, gene, gene>& regulation,
    active_support<gene, gene>& active_edges,
    axis<gene> genes,
    ce::trajectory trajectory)
<[
    given ce::profileof(expression)
        matches ce::bio::activated_fibroblast;

    given ce::persists(
        structure(regulation),
        ce::across(trajectory));

    given ce::persists(
        orderof(expression),
        ce::across(trajectory));

    given ce::changes(
        values(regulation),
        ce::every(trajectory.step()));

    given ce::support_evolves(
        active_edges,
        ce::every(trajectory.step()));

    given ce::uses(regulation)
        == trajectory.size();

    prefer ce::minimum_amortized_latency;

    require ce::deterministic;
    require ce::no_host_synchronization;

    offer decomposition by_module;

    inspect semantics, state_flow;
    inspect candidates, costs, persistence;
::
    for (auto step : trajectory) {
        ce::update_active_support(
            active_edges,
            ce::bio::receptor_activity(expression));

        expression =
            expression
            -[regulation where active_edges]->
            genes;

        update_regulatory_values(
            values(regulation),
            expression,
            step);
    }
]>

} // namespace cardiac
```

Mechanically, Cellerator can:

1. validate domain, axis, structure, support, generation, and output contracts;
2. bind the activated-fibroblast profile;
3. treat topology and order as persistent;
4. treat values and active support as dynamic generations;
5. compare unsplit and module decompositions;
6. require exact contribution ownership and a valid partial sum;
7. consider candidates that share structure-only projections;
8. price per-step mask and value updates;
9. reject nondeterministic plans;
10. build one recurrent prepared stage graph;
11. publish each successful generation without a host wait;
12. explain every reuse and invalidation decision.

The code remains readable at the semantic level. The expert decomposition is declared separately and only offered.

## 27. Common mistakes

### Mistake: treating equal shape as equal biology

```cpp
// Wrong assumption: both axes have 20,000 entries.
ce::unsafe_reinterpret_axis(cell_axis, gene_axis);
```

Use an explicit relation or a verified mapping.

### Mistake: calling every support change structural

Toggling activity over stable edges should use an active-support overlay. Rebuilding structure wastes identity and projection reuse.

### Mistake: hiding overwrite versus accumulation

Use `=` or `+=`. Do not bury destination initialization in a candidate.

### Mistake: canonicalizing every intermediate

Canonicalize at an observer boundary or when a requirement demands it. Let connected fields preserve useful orders.

### Mistake: using `require` for a preference

```cpp
require ce::minimum_latency; // conceptually wrong
```

Latency is an objective, not a semantic predicate. Use `prefer`.

### Mistake: forcing before reading the report

Offer your candidate first. Learn why it loses. Then fix the fact, cost model, or candidate.

### Mistake: lying in an effect contract

An incorrect effect contract can invalidate planning. Use `opaque` when uncertain.

### Mistake: parsing textual IR

Use the typed IR API. Dumps are for humans.

### Mistake: assuming the field launches one kernel

A field is a planning envelope. Inspect the executable IR when launch count matters.

### Mistake: making physical padding biological

Padding and alignment gaps have no logical edge identity and never own contributions.



## 28. Operation-family cookbook

The relation arrow is iconic, but Cellerator's operation algebra is broader.

### 28.1 Relation apply

```cpp
response =
    expression -[regulation]-> genes;
```

Use for typed transfer across biological connectivity.

### 28.2 Transpose apply

```cpp
source_gradient +=
    destination_gradient
    -[transpose(regulation)]->
    source_genes;
```

Use when source and destination roles reverse while logical edge identity remains stable.

### 28.3 Support-restricted contraction

```cpp
edge_scores =
    ce::contract_on(
        supportof(regulation),
        expression,
        target_state,
        ce::dot);
```

Use when the result lives on exact logical edges.

### 28.4 Segment reduction

```cpp
tissue_expression =
    ce::segment_reduce(
        expression,
        tissue_segments,
        ce::sum);
```

Use for exact grouped aggregation.

### 28.5 Segment normalization

```cpp
incoming_attention =
    ce::segment_normalize(
        edge_scores,
        incoming_edge_segments,
        ce::softmax);
```

Use for segment-local normalization with explicit empty, singleton, nonfinite, and backward semantics.

### 28.6 Edge map or gate

```cpp
scaled_relation =
    ce::edge_map(
        regulation,
        dosage,
        ce::multiply);

gated_relation =
    ce::edge_gate(
        regulation,
        receptor_state);
```

Use when logical edge identity is preserved while values or activity are transformed.

### 28.7 Sparse axis update

```cpp
ce::sparse_update(
    expression,
    perturbed_genes,
    perturbation_delta,
    ce::add);
```

Use for explicit sparse mutation of state or values.

### 28.8 Bundle

```cpp
combined =
    ce::bundle(
        regulation,
        signaling,
        cell_contact);

response =
    expression -[combined]-> genes;
```

Use when several relations participate in one grouped apply while retaining their identities.

### 28.9 Hierarchy pool and broadcast

```cpp
module_state =
    ce::pool(
        gene_state,
        gene_modules,
        ce::mean);

gene_context =
    ce::broadcast(
        module_state,
        gene_modules);
```

Use when the source semantics are a hierarchy rather than an anonymous sparse matrix.

### 28.10 Coupled moments

```cpp
auto [mean_signal, signal_energy] =
    ce::relation_moments(
        expression,
        regulation,
        genes);
```

Use when multiple exact outputs can share a traversal but remain separate semantic results.

These are compiler-semantic library operations. New operation families can join through the operation protocol without demanding new punctuation.

## 29. A productive optimization workflow

A good Cellerator optimization session should look like compiler engineering, not guesswork.

### Step 1: write the semantic program

```cpp
<[
    response =
        expression -[regulation]-> genes;
]>
```

Confirm correctness and semantic typing first.

### Step 2: inspect semantic understanding

```cpp
inspect semantics, state_flow;
```

Check axes, relation identity, support, generations, output effects, and numerical policy.

### Step 3: provide real workload facts

```cpp
given ce::uses(regulation) >= 10000;
given ce::persists(
    structure(regulation),
    ce::across(trajectory));
```

Do not begin with a preferred kernel.

### Step 4: inspect candidate and complete cost reports

```cpp
inspect candidates, costs, persistence;
```

Find the actual dominant phase.

### Step 5: improve the right layer

If packing dominates, improve value ownership or reuse.

If an order transform dominates, join fields or preserve order.

If an opaque call dominates, add an effect contract.

If the candidate set is weak, offer a candidate.

If decomposition is poor, offer an exact decomposition.

If the cost model lacks global information, offer external cost.

### Step 6: offer before forcing

Let your idea compete. Read why it wins or loses.

### Step 7: force only for a reason

Force for experiments, reproducibility, deployment lock-down, or knowledge outside the planner.

### Step 8: inspect lower IR

```cpp
inspect ir<projection>, ir<packed>, ir<executable>, ir<native>;
```

Confirm that the intended mechanism reached code generation.

This workflow is how Cellerator stays both automatic and accountable.

## 30. Building Cellerator libraries

### 30.1 Put semantics in interfaces

A reusable operation should expose:

- typed domains and axes;
- exact relation or support contracts;
- output effects;
- numerical and determinism requirements;
- data-state transfer;
- lifetime effects;
- semantic IR lowering;
- candidate discovery.

Do not make the compiler reverse-engineer these facts from pointer arithmetic.

### 30.2 Keep construction in the library

A library can provide friendly owners and builders:

```cpp
ce::bio::regulatory_network<float> network =
    ce::bio::load_regulatory_network<float>(
        source,
        genes);
```

The object should expose the same low-level structure and value views:

```cpp
relation<float, gene, gene> regulation =
    network.relation();

relation_structure<gene, gene> topology =
    network.structure();

relation_values<float, gene, gene> weights =
    network.values();
```

Convenience should be peelable.

### 30.3 Publish source-level candidates

A candidate provider should be linked or included before planning. Its catalog entries should be immutable and versioned.

Avoid a hot-path registry that scans plugins or allocates on first use.

### 30.4 Give every abstraction an exit

A useful library type should let advanced users reach:

- non-owning semantic views;
- persistent identities;
- axes and orders;
- structure epoch and value generation;
- exact coverage;
- candidate or decomposition hooks;
- raw storage when ownership permits.

A convenience wrapper that traps the programmer above the compiler is not a good Cellerator abstraction.

### 30.5 Preserve C++ culture

Templates, concepts, overloads, constexpr construction, RAII, custom allocators, and source inclusion are strengths.

Use Cellerator syntax where it communicates semantic information that ordinary C++ cannot. Do not rename ordinary arithmetic or memory management with biological vocabulary.

## 31. Compilation and artifact workflow

A practical toolchain can support several routes without changing source meaning.

### Compile from representative data

```text
source + profile + target
    -> semantic geometry
    -> exact plan
    -> executable
```

### Reuse semantic geometry

```text
source + compatible semantic artifact + target
    -> target cover and projection
    -> executable
```

### Reuse a target-specific execution artifact

```text
source + compatible executable artifact
    -> validate dependencies
    -> bind values and pointers
    -> launch
```

### Resume after partial invalidation

```text
value generation changed
    -> reuse structure, geometry, cover, projection
    -> refresh packed values

target changed
    -> reuse semantic layers
    -> rebuild projection and executable

structure epoch changed
    -> return to structural evidence and exact coverage
```

The source can offer artifacts, but loading files and managing stores belongs to the compiler driver or standard library, not the core grammar.

An artifact report should expose:

```text
artifact: cardiac.regulation.sm70.v14
accepted through: physical projection
replayed: packed operand, executable recipe
reason: value generation 742 -> 743
```

## 32. Returning to ordinary C++

A field ends at `]>`.

```cpp
<[
    response =
        expression -[regulation]-> genes;
]>

ce::await(response);

std::span<const float> host_view =
    copy_to_host(response);

analyze_with_existing_cpp(host_view);
```

Outside the field:

- C++ evaluation rules apply;
- no new joint planning scope is implied;
- Cellerator views remain typed values;
- you may open another field later;
- you may call a named field as a normal function;
- you may ignore Cellerator completely for the next thousand lines.

This is not an all-or-nothing programming environment. Cellerator is a precise semantic instrument embedded in a broad systems language.

## 33. Performance philosophy in practice

Cellerator's default ambition is not "find a decent kernel." It is "find the best complete executable path the available evidence can justify."

That means asking questions at several scales:

- Is the semantic operation correct?
- Can operations share a traversal?
- Can a persistent order remove a transition?
- Is decomposition worth its halo or partial merge?
- Does a packed representation amortize?
- Are dynamic values cheaper to repack or consume directly?
- Does a matrix-engine region plus residual beat pure sparse?
- Is a multi-extent direct launch cheaper than assembly?
- Is graph capture worth the constraints?
- Does global communication reverse the local winner?
- Is the measurement current and uncontaminated?

The fastest kernel can lose the complete plan. The prettiest decomposition can lose to an unsplit fallback. A manual candidate can beat every built-in. A compiler-selected candidate can beat the author's favorite.

The language is designed to let all four facts coexist without drama.

## 34. Reading the two documents together

Use the language specification when you need to know:

- what a construct means;
- what the compiler must preserve;
- whether a behavior is an error, warning, preference, or implementation choice;
- what stability an IR level receives;
- where the language/library boundary sits.

Use this guide when you need to know:

- how to express a computation;
- which control level to use;
- what the compiler does mechanically;
- how to diagnose a plan;
- how to descend into decomposition or IR;
- when to leave Cellerator and write manual code.

The most important habit is simple:

> State what is true about the biology and the workload before stating how the machine must execute it.

Then inspect the compiler's answer. Take over where you know better.

## 35. Source grounding

The programming experience described here was designed around concrete Cellerator machinery, especially:

- typed operation problems and relation algebra;
- independent structure epochs and value generations;
- logical edge order and explicit output effects;
- workload profiles and reuse horizons;
- semantic geometry and target cover separation;
- exact decomposition, halos, replicas, and partial algebra;
- atom requirements, affordances, planes, and multi-extent binding;
- complete end-to-end and connected transition cost;
- provider-linked candidate catalogs;
- projection and packed-value separation;
- prepared stage graphs and graph capture;
- generation readiness and publication;
- staged lowering resumption;
- external cost and bounded candidate exchange.

The guide intentionally does not map every current C++ record to a keyword. It exposes the stable concepts that make those records necessary.

That is the proposed Cellerator experience: write explicit biological computation, give the compiler the truths it cannot infer, examine its reasoning, and keep every door unlocked.
