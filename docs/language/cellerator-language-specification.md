# Cellerator Programming Language: Proposed Specification

**Status:** First serious language-design proposal
**Proposed language revision:** 0.1
**Research baseline:** Cellerator `main` at `8a56e78a367450d67f6b06bf450279de8379793f`, inspected on 2026-09-01
**Normative status:** Design specification for future implementation, not a description of an implemented frontend

**Companion document:** [cellerator-programming-guide.md](cellerator-programming-guide.md)

## Contents

- [1. Purpose and scope](#1-purpose-and-scope)
- [2. Normative vocabulary](#2-normative-vocabulary)
- [3. Design principles](#3-design-principles)
- [4. Relationship to C++](#4-relationship-to-c)
- [5. Lexical and grammatical additions](#5-lexical-and-grammatical-additions)
- [6. Compiler-semantic types](#6-compiler-semantic-types)
- [7. Biological typing and explicit escape hatches](#7-biological-typing-and-explicit-escape-hatches)
- [8. Relation transfer expressions](#8-relation-transfer-expressions)
- [9. Operation families](#9-operation-families)
- [10. Execution fields](#10-execution-fields)
- [11. Planning directives and authority hierarchy](#11-planning-directives-and-authority-hierarchy)
- [12. Representative data and data-state evolution](#12-representative-data-and-data-state-evolution)
- [13. Persistence, reuse, identity, and biological time](#13-persistence-reuse-identity-and-biological-time)
- [14. Numerical, determinism, output, and order contracts](#14-numerical-determinism-output-and-order-contracts)
- [15. C, C++, CUDA, and native interoperability](#15-c-c-cuda-and-native-interoperability)
- [16. Exact coverage, decomposition, atoms, and extents](#16-exact-coverage-decomposition-atoms-and-extents)
- [17. Custom candidates, cost models, and forced realization](#17-custom-candidates-cost-models-and-forced-realization)
- [18. Intermediate representation as a programming feature](#18-intermediate-representation-as-a-programming-feature)
- [19. Diagnostics and introspection](#19-diagnostics-and-introspection)
- [20. Errors, warnings, and fallback](#20-errors-warnings-and-fallback)
- [21. Compilation model](#21-compilation-model)
- [22. Standard-library boundary](#22-standard-library-boundary)
- [23. Compact grammar sketch](#23-compact-grammar-sketch)
- [24. Integrated examples](#24-integrated-examples)
- [25. Implementation-defined behavior](#25-implementation-defined-behavior)
- [26. Extension and versioning strategy](#26-extension-and-versioning-strategy)
- [27. Rejected or avoided designs](#27-rejected-or-avoided-designs)
- [28. Open Design Questions](#28-open-design-questions)
- [29. Research grounding](#29-research-grounding)

## 1. Purpose and scope

This document proposes the first coherent programming-language specification for Cellerator.

Cellerator is a low-level, explicit, performance-first biological programming language implemented as a C++ language extension. It lets programmers describe biological computation while preserving enough semantic structure for the compiler to select physical representations, decompositions, projections, packed operands, candidate implementations, and complete execution paths from representative data and target properties.

Cellerator is not a high-level workflow language. It does not replace C++ control, templates, pointers, custom allocation, CUDA, native libraries, or manual kernels. It adds a small semantic layer where ordinary C++ cannot communicate what biological objects mean.

The governing design rule is:

> Cellerator syntax exposes facts that participate in semantic validation, data-state reasoning, or physical planning. Constructions that merely make those facts convenient belong in the Cellerator standard library.

The ordinary path is intentionally short:

```cpp
#pragma cellerator

<[
    response = expression -[regulation]-> target_genes;
]>
```

The expert path remains open:

```cpp
#pragma cellerator

field void propagate(...) <[
    given ce::persists(structure(regulation), trajectory);
    prefer ce::minimum_latency;
    offer decomposition by_module;
    offer candidate my_sparse_mma_hybrid;
    inspect semantics, candidates, costs, ir<projection>;
::
    response = expression -[regulation]-> target_genes;
]>
```

Both are the same language. There is no separate "advanced Cellerator."

## 2. Normative vocabulary

The words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** express language requirements.

- **MUST** and **MUST NOT** define required language behavior.
- **SHOULD** and **SHOULD NOT** define strong defaults that an implementation may depart from only for a documented reason.
- **MAY** defines permitted behavior.
- **Implementation-defined** behavior MUST be documented and queryable.
- **Unspecified** behavior need not be documented, but MUST remain within stated semantic constraints.
- **Provisional** syntax is part of this proposal but is explicitly expected to receive further design pressure before implementation.

## 3. Design principles

### 3.1 Biological meaning must survive lowering

Domains, axes, relations, exact logical support, ordering, structure identity, structure epochs, value generations, output effects, and numerical contracts MUST remain available until the compiler has made the decisions that depend on them.

Equal dimensions, equal pointer values, or compatible physical shapes MUST NOT establish biological equivalence.

### 3.2 Physical realization is deliberately plural

One semantic relation operation MAY become:

- one conventional sparse kernel;
- one library call;
- multiple exact sparse fragments;
- a matrix-engine region plus an exact residual;
- a relation bundle or chain;
- a set of independently prepared atoms;
- a multi-extent direct launch;
- a packed operand plus a prepared stage graph;
- a target-specific program selected from measured alternatives.

One source statement is not required to correspond to one kernel, one launch, one allocation, or one physical representation.

### 3.3 Exactness precedes optimization

Approximate profiles, sampled support, affinity, recurrence, and candidate discovery MAY propose implementations. They MUST NOT establish execution correctness.

Every selected realization MUST preserve exact logical coverage, output semantics, numerical requirements, and contribution ownership. Overlapping contributions require a declared and verified partial-result algebra.

### 3.4 Automatic optimization is default, not authority without appeal

With no constraints, the compiler SHOULD choose the fastest correct complete realization it can establish for the supplied target and representative data.

A programmer MUST be able to:

1. give planning facts;
2. express preferences;
3. impose hard constraints;
4. offer custom transformations, decompositions, candidates, and cost models;
5. force a validated realization;
6. leave Cellerator planning and execute fully manual C++ or CUDA.

### 3.5 Progressive exposure

Ordinary programs SHOULD not need to mention atoms, projections, packing, target covers, or IR.

Expert programs MUST be able to inspect and manipulate those layers through typed, versioned interfaces. The expert surface SHOULD feel like programming against a compiler, not serializing private structs or writing compiler assembly.

### 3.6 No hidden hot-path work

Unless explicitly requested by source semantics, an already prepared field MUST NOT perform hidden:

- candidate discovery;
- geometry search;
- catalog parsing;
- global sorting;
- allocation;
- host synchronization;
- device selection;
- implicit canonicalization;
- structure hashing;
- runtime topology search.

Preparation, packing, transfer, canonicalization, synchronization, and communication MUST be visible to the planner and to diagnostics.

### 3.7 Biology-first without a mandatory ontology

Cellerator MUST support nominal biological domain types such as genes, cells, loci, enhancers, reads, modules, populations, or trajectory positions.

The language MUST NOT hard-code one biological ontology. Users may define new domain types, erase domain distinctions explicitly, and provide custom relations among them.

## 4. Relationship to C++

### 4.1 C++ is the host language

A Cellerator source file is a C++ source file with Cellerator extensions enabled for that physical file. Except where this specification states otherwise:

- C++ lexical rules apply;
- preprocessing applies;
- declarations, namespaces, templates, overloads, concepts, exceptions, pointers, references, classes, and control flow retain C++ meaning;
- ordinary expressions are compiled by the host C++ implementation;
- CUDA language features remain available when the selected toolchain supports them.

Cellerator does not reserve a different filename extension as its semantic switch.

### 4.2 Source-file opt-in

The directive:

```cpp
#pragma cellerator
```

enables Cellerator grammar and semantic analysis from the directive to the end of the current physical source file.

A version-pinned form is proposed:

```cpp
#pragma cellerator 0.1
```

The unversioned form selects the implementation's default supported revision. Production libraries SHOULD pin a revision once revisions become stable.

The mode:

- MUST end automatically at the physical file boundary;
- MUST NOT leak from an included file into its includer;
- MUST NOT leak from an includer into an included file unless that included file opts in itself;
- MUST restore the prior mode when preprocessing returns from an include;
- MUST NOT itself open an execution field or authorize data-adaptive planning.

An implementation SHOULD expose:

```cpp
__has_feature(cellerator)
__has_cellerator_feature(feature_name)
CELLERATOR_LANGUAGE_REVISION
```

or equivalent feature-test facilities.

**Rationale.** A pragma is the least surprising way to tell a Clang-like frontend that subsequent tokens use an extended grammar. It preserves ordinary C++ build style and avoids relying on filenames. A paired `end cellerator` directive is unnecessary because the physical file is the natural parsing boundary.

### 4.3 Unsupported compilers

A compiler that does not implement Cellerator will ordinarily reject Cellerator grammar. Cellerator constructs that alter types or semantics MUST NOT be expressed only as ignorable C++ attributes.

Libraries MAY use preprocessor feature tests to provide ordinary C++ fallbacks.

## 5. Lexical and grammatical additions

Cellerator revision 0.1 proposes the following primary additions:

- `domain` declarations;
- compiler-semantic type constructors;
- relation transfer syntax `-[ ... ]->`;
- anonymous execution fields `<[ ... ]>`;
- named `field` functions;
- field directives `given`, `prefer`, `require`, `offer`, `force`, and `inspect`;
- program-point directives `expect` and `verify`;
- trailing `effects(...)` contracts;
- typed `ir<level>` reflection;
- `transform` functions for compile-time IR transformation;
- contextual planning relations such as `matches` and `in` inside planning, `expect`, and `verify` clauses.

These words are contextual keywords. Outside a Cellerator-enabled physical file, they have no Cellerator meaning. Within a Cellerator-enabled file, they retain ordinary identifier meaning where the grammar is unambiguous.

The delimiters `<[` and `]>` are indivisible Cellerator tokens while Cellerator mode is active.

## 6. Compiler-semantic types

### 6.1 General rule

A compiler-semantic type is a source-level type whose values carry contracts used by Cellerator semantic analysis or planning.

A compiler-semantic type does not imply ownership, allocation, or a particular physical layout. The standard library supplies owners, views, binders, builders, adapters, and containers.

Revision 0.1 proposes these core type families:

```cpp
domain D;

axis<D>
order<D>
state<T, D...>

relation_structure<Source, Destination>
relation_values<T, Source, Destination>
relation<T, Source, Destination>

support<Source, Destination>
active_support<Source, Destination>
segments<D>

profile
```

Expert protocols additionally use:

```cpp
coverage
partial_algebra
decomposition
atom
extent
projection
candidate
cost_model
realization
ir<Level>
```

The compiler MAY implement these as intrinsic types, compiler-recognized library types, or a hybrid. Their observable semantics MUST follow this specification.

### 6.2 Domains

A declaration such as:

```cpp
domain gene;
domain cell;
domain enhancer;
domain regulatory_module;
```

declares a nominal domain type.

Two separately declared domains are distinct even when their runtime extents match.

Domains MAY be declared inside namespaces:

```cpp
namespace organism {
    domain gene;
    domain cell;
}
```

The standard library SHOULD provide common biological domain declarations under `cellerator::bio`, but those declarations carry no privileged causal or ontological authority.

A user MAY define an abstract domain:

```cpp
domain latent_feature;
```

A relation is the explicit bridge between distinct domains. There is no implicit conversion merely because two domains contain the same number of entities.

### 6.3 Axes

An `axis<D>` is a non-owning semantic identity for positions in domain `D`. It includes or resolves:

- persistent domain identity;
- persistent order identity;
- geometry identity when relevant;
- partition identity when relevant;
- extent;
- optional hierarchy membership;
- a stable relationship to canonical identity.

An axis value does not own its labels, maps, or data storage unless a standard-library owner says otherwise.

Two axes are compatible only when their domain type and required persistent identities are compatible. Equal extent is insufficient.

The language provides intrinsic queries:

```cpp
domainof(axis_value)
orderof(axis_value)
identityof(axis_value)
extentof(axis_value)
partitionof(axis_value)
```

The exact return types are compiler-semantic identity types.

### 6.4 Orders

An `order<D>` names a persistent ordering of a domain. Orders are semantic because:

- relation edge values are indexed in logical edge order;
- prepared projections may use a different physical order;
- connected operations may avoid a transform when producer and consumer orders agree;
- canonicalization, permutation, packing, and recovery carry measurable cost.

The compiler MUST NOT silently treat one order as another.

Explicit operations such as `reorder`, `canonicalize`, and order-preserving view construction are supplied by the standard library and lowered as costed semantic operations.

### 6.5 States

A `state<T, D...>` is a non-owning typed view of quantitative biological state over one or more axes.

For example:

```cpp
state<float, cell, gene>
state<std::uint8_t, cell>
state<float, trajectory_point, cell, gene>
```

A state value carries or resolves:

- one exact axis per domain parameter;
- numerical storage policy;
- logical value type;
- value generation;
- residency and readiness at launch time;
- mutability;
- optional quantization;
- optional dirty extents;
- an ownership or aliasing contract.

Declaring a state MUST NOT allocate memory. A standard-library owner or binder must provide storage explicitly.

The standard library SHOULD distinguish owning storage from non-owning semantic views, for example:

```cpp
ce::state_buffer<float, cell, gene> owned;
state<float, cell, gene> view = owned.view(cells, genes);
```

### 6.6 Relation structures and values

A `relation_structure<S, D>` describes immutable logical connectivity from source domain `S` to destination domain `D`.

It carries or resolves:

- stable structure identity;
- structure epoch;
- source axis;
- destination axis;
- exact logical support;
- stable logical edge identity;
- logical edge order;
- projection catalog identity;
- partition and hierarchy information where present.

A `relation_values<T, S, D>` describes mutable values attached to the logical edges of a compatible relation structure. It carries or resolves:

- value generation;
- storage and logical numeric policy;
- value ownership mode;
- logical-edge or projection-local layout;
- optional quantization;
- optional composite physical value planes;
- readiness and residency.

A bound `relation<T, S, D>` pairs one relation structure with one compatible value generation.

Example:

```cpp
relation_structure<gene, gene> topology =
    ce::bind_structure<gene, gene>(...);

relation_values<float, gene, gene> weights =
    ce::bind_values<float>(topology, ...);

relation<float, gene, gene> regulation =
    ce::bind_relation(topology, weights);
```

The standard library MAY provide a single convenience binder for ordinary use.

A pointer change MUST NOT change relation identity. A value-generation change MUST NOT by itself change structure identity or structure epoch.

### 6.7 Support

A `support<S, D>` names exact logical membership over a relation's edge identity space.

An `active_support<S, D>` is a mutable generation-tagged overlay over an immutable relation edge set. Clearing an active-support bit does not delete the underlying logical edge and does not change structure identity.

Structural support mutation, meaning addition, deletion, or reidentification of underlying logical edges, advances the structure epoch.

Support compatibility requires matching:

- structure identity;
- structure epoch;
- source axis;
- destination axis;
- logical edge order.

### 6.8 Segments and groupings

A `segments<D>` value identifies a partition of an axis into ordered or otherwise explicitly described segments. Segment identity participates in reductions, normalization, decomposition, hierarchy pooling, and partial-result reconstruction.

A segmentation is not inferred from equal sizes or adjacency unless its constructor explicitly defines that interpretation.

### 6.9 Profiles

A `profile` is a compiler-facing description of expected data state and workload, not a runtime data container and not a biological causal claim.

A profile may include:

- expected support statistics;
- degree and occupancy distributions;
- dense width ranges;
- activity and prevalence;
- co-support or affinity evidence;
- value dynamics;
- reuse horizons;
- operation frequencies;
- memory constraints;
- target-independent semantic geometry;
- a bounded set of alternative states;
- confidence and evidence revision.

Profiles are normally supplied to compilation externally. Source code refers to symbolic profile objects rather than embedding dataset paths or workflow logic.

## 7. Biological typing and explicit escape hatches

### 7.1 Static domain typing

The compiler MUST reject a semantic operation when its domain types are incompatible.

For example, this is ill-typed unless a compatible relation is explicitly supplied:

```cpp
state<float, cell> cells;
axis<gene> genes;

// Error: a cell state is not a gene state.
auto invalid = cells -[some_gene_relation]-> genes;
```

Static domain typing prevents broad classes of accidental axis substitution while remaining open to user-defined domains.

### 7.2 Runtime identity validation

Static domain compatibility does not prove exact axis compatibility. Prepared and launched operations MUST validate the persistent identities required by their contracts.

The compiler or runtime MUST reject stale:

- structure epochs;
- value generations;
- order identities;
- projection identities;
- readiness generations;
- partition or hierarchy identities when required.

### 7.3 Domain erasure

Low-level generic code MAY explicitly erase nominal domain information:

```cpp
axis<any_domain> erased = ce::erase_domain(genes);
```

Domain erasure disables some static checking but does not erase persistent runtime identity.

An unchecked domain reinterpretation MUST require an explicitly unsafe standard-library facility. It MUST NOT be an implicit cast.

**Rationale.** The type system should stop meaningful mistakes without becoming a biological ontology or a cage. Nominal domains provide a strong default. Explicit erasure preserves systems-level freedom.


## 8. Relation transfer expressions

### 8.1 Purpose

The relation transfer expression is the primary notation for moving or accumulating quantitative state through typed biological connectivity.

Its proposed form is:

```cpp
result = source -[relation_selector]-> destination_axis;
```

For example:

```cpp
state<float, cell, gene> response;
response = expression -[regulation]-> target_genes;
```

The expression says:

1. select one source axis of `source`;
2. apply the exact logical relation named by `regulation`;
3. replace that source axis with the relation's destination axis;
4. produce a state whose remaining axes are preserved;
5. apply the assignment operator's output-update semantics.

The expression does not select CSR, COO, blocked sparse storage, a matrix engine, a single kernel, a launch geometry, or a device.

### 8.2 Static typing

For a relation:

```cpp
relation<W, S, D> r;
```

and an input:

```cpp
state<T, P..., S, Q...> x;
```

the expression:

```cpp
x -[r]-> destination_axis
```

has a result compatible with:

```cpp
state<R, P..., D, Q...>
```

where `R` follows the operation's numerical policy.

The right operand MUST be an axis compatible with the destination axis of the relation. This makes the destination of the biological transfer explicit in source even when the relation object already contains a bound destination axis.

If exactly one source axis of `x` is compatible with `r`, that axis is selected. If more than one axis is compatible, the programmer MUST disambiguate:

```cpp
response = pair_state -[regulation on left_genes]-> target_genes;
```

The `on` clause is part of the relation selector, not a physical layout annotation.

### 8.3 Relation selectors

The revision 0.1 relation selector supports the conceptual form:

```text
relation-expression
    [ on source-axis-expression ]
    [ where support-expression ]
```

Examples:

```cpp
response = expression -[regulation]-> genes;

response = paired_expression
         -[regulation on regulator_genes]->
         target_genes;

response = expression
         -[regulation where active_edges]->
         target_genes;
```

A `where` clause binds exact support or a compatible active-support overlay. It MUST preserve the underlying logical edge identity and order required by the relation.

A structural subset that creates a new relation identity is not merely a `where` overlay. It must be constructed explicitly as a new relation structure or exact coverage view.

### 8.4 Orientation

Orientation is semantic and MUST be explicit when it is not the relation's forward orientation.

The proposed form is:

```cpp
source_gradient =
    destination_gradient -[transpose(regulation)]-> source_genes;
```

`transpose(relation)` is a compiler-semantic view. It reverses the source and destination axes while preserving the stable identity of each logical edge and the relationship between forward and transpose value positions.

The compiler MAY select a stored transpose projection, construct one, reuse a persistent one, or execute a direct transpose candidate. It MUST charge any required remapping, construction, packing, or communication cost.

### 8.5 Output-update semantics

A relation transfer is an expression. The surrounding assignment operator states how the destination is updated:

```cpp
response  = expression -[regulation]-> genes;  // overwrite
response += expression -[regulation]-> genes;  // accumulate
```

Ordinary compound assignments have their ordinary C++ meaning, subject to numerical and aliasing contracts.

Affine accumulation and partial writes use explicit standard-library lvalue adapters:

```cpp
ce::affine(response, alpha, beta) =
    expression -[regulation]-> genes;

ce::partial(response, exact_gene_coverage, partial_sum) =
    expression -[regulation]-> genes;
```

An implementation MUST NOT silently convert overwrite into accumulation, silently zero an accumulated destination, or silently canonicalize an output.

Aliasing between source, relation values, and destination is legal only when the operation contract and selected candidate both permit it.

### 8.6 Chaining

Relation transfers associate from left to right:

```cpp
response =
    expression
    -[gene_to_module]-> modules
    -[module_to_gene]-> genes;
```

The chained form describes two semantic operations and one intermediate state. It does not require an intermediate allocation.

Within one execution field, the compiler MAY:

- materialize the intermediate;
- keep it in a persistent noncanonical order;
- fuse compatible stages;
- use a relation-chain candidate;
- split either relation;
- choose distinct physical candidates for the two stages.

The compiler MUST preserve the intermediate domain, extent, output effects, and any observable use of the intermediate.

### 8.7 Value-less and structural relations

A relation may have implicit unit values, predicate values, or another declared value interpretation. Such interpretation MUST be part of its type or semantic contract.

A bare structure cannot silently acquire numerical weights. Binding values to a structure is explicit:

```cpp
auto regulation = ce::with_values(regulatory_topology, regulatory_weights);
```

### 8.8 Expression lifetime

A relation transfer expression is a semantic operation expression. Outside an execution field, it is ill-formed in revision 0.1 unless it appears in a named field body.

This restriction prevents a source expression from looking like ordinary C++ while silently requesting a disconnected one-operation planner invocation. A future revision may permit an implicit single-operation field if experience shows that it is useful and unambiguous.

**Rationale.** The arrow is reserved for Cellerator proper. It communicates biological transfer while assignment communicates mutation. This leaves advanced packing, decomposition, and target mechanics available without making them part of the ordinary expression.

## 9. Operation families

### 9.1 General rule

The relation arrow is not a universal operator for every operation Cellerator can plan.

The compiler currently distinguishes operations whose semantics are not naturally "state travels from a source domain to a destination domain." Revision 0.1 therefore defines a small set of semantic operation families.

Only relation transfer receives dedicated infix notation. The remaining families are compiler-semantic standard-library intrinsics. Their implementations are ordinary source-level Cellerator library code plus compiler-visible contracts, not necessarily precompiled functions.

### 9.2 Relation transfer

Relation transfer includes:

- forward relation apply;
- transpose relation apply;
- relation bundles with a shared destination;
- relation chains;
- relation exchange across a declared partition or hierarchy boundary.

The simple and chained arrow forms cover the ordinary cases. Bundle and exchange construction is supplied by the standard library.

```cpp
auto combined = ce::bundle(regulation, signaling, contact);
response = expression -[combined]-> genes;
```

A bundle preserves the identities of its member relations. It does not flatten them into one anonymous matrix.

### 9.3 Support-local contraction

A support-local contraction computes values on the exact logical edges of a relation or coverage. Typical uses include edge scoring, gradient formation, and support-restricted interaction terms.

```cpp
edge_scores = ce::contract_on(
    supportof(regulation),
    source_state,
    destination_state,
    ce::dot);
```

The result is indexed by stable logical edge identity unless an explicitly physical view is requested.

The compiler MAY lower this to SDDMM-like code, a fused edge stage, a segmented traversal, or another exact candidate.

### 9.4 Segmented reduction and normalization

Segmented operations act over a declared segment space:

```cpp
tissue_sum = ce::segment_reduce(
    expression, tissues, ce::sum);

attention = ce::segment_normalize(
    edge_scores, incoming_edges, ce::softmax);
```

Supported semantic families include sum, maximum, log-sum-exp, softmax, log-softmax, L1, L2, RMS normalization, and their declared backward operations.

Empty-segment and singleton behavior MUST be defined by the operation contract. It is not implementation folklore.

### 9.5 Edge maps, gates, and active support

Edge-local maps and gates preserve logical edge identity unless they explicitly construct a new relation:

```cpp
scaled = ce::edge_map(regulation, ce::multiply, dosage);

gated = ce::edge_gate(regulation, receptor_state);

ce::update_active_support(
    active_edges, predicate, ce::bit_mask);
```

A mutable active-support mask is an overlay over stable topology. It advances its own generation and does not by itself advance the structure epoch.

A predicate that actually inserts or deletes logical edges is structural mutation and must be expressed as such.

### 9.6 Sparse axis updates

Sparse updates mutate selected positions on a typed axis:

```cpp
ce::sparse_update(
    expression,
    perturbed_genes,
    delta,
    ce::add);
```

The update operation MUST declare whether indices are unique, whether they are in persistent order, and whether canonical identity is preserved.

The compiler may exploit these facts but MUST retain C++-visible update semantics.

### 9.7 Hierarchy pool and broadcast

Hierarchy operations move state between explicit parent and child domains without manufacturing an adjacency matrix:

```cpp
module_state = ce::pool(
    gene_state, gene_modules, ce::mean);

gene_context = ce::broadcast(
    module_state, gene_modules);
```

The hierarchy and mapping identities are semantic inputs. A physical implementation may use direct segment traversal, a relation projection, or a custom candidate.

### 9.8 Moments and coupled traversals

Operations that compute multiple exact functionals of one relation traversal may be represented as separate semantic results:

```cpp
auto [first, second] =
    ce::relation_moments(expression, regulation, genes);
```

The compiler may select a paired traversal when profitable. The source does not promise fusion merely because results are written together.

### 9.9 Composition descriptors

Advanced users may construct explicit semantic compositions:

```cpp
auto composition = ce::compose(
    ce::stage(edge_scores),
    ce::stage(attention),
    ce::stage(response));
```

A composition describes dependencies and admissible fusion. It does not itself select a physical kernel.

### 9.10 Extensibility

A library-defined operation may participate in Cellerator planning if it implements the versioned semantic-operation protocol:

- stable operation identity;
- typed inputs and outputs;
- exact semantic effects;
- numerical and determinism requirements;
- data-state transfer function;
- candidate and decomposition discovery hooks;
- IR lowering hooks where applicable.

The language MUST NOT require every future operation family to become a keyword or operator.

## 10. Execution fields

### 10.1 Core meaning

An execution field is a Cellerator planning envelope.

The anonymous form is:

```cpp
<[
    statements
]>
```

The form with a planning prologue is:

```cpp
<[
    planning-directives
::
    statements
]>
```

The `::` delimiter separates compile-time planning declarations from executable statements. It is omitted when there is no prologue.

An execution field states:

> The enclosed statements form one semantic biological computation. Cellerator may select and compose correct physical realizations across the field, subject to C++ observable behavior and the field's declared facts, preferences, constraints, offers, and forced choices.

A field is not a GPU region, a parallel-for region, a storage declaration, or a promise of one launch.

### 10.2 Field graph

The compiler MUST construct a semantic dependence graph for each outermost field.

The graph includes:

- compiler-semantic operations;
- named field calls whose semantic bodies are available;
- explicit order, generation, readiness, and mutation transitions;
- effect-contracted C/C++ calls;
- opaque barriers for calls or constructs whose effects are unknown;
- C++ control and side-effect dependencies that constrain movement.

The compiler MAY reorder, fuse, split, pack, preserve noncanonical order, or share preparation only when the graph proves the transformation legal.

### 10.3 Optimization visibility

Two operations in different outermost fields are not jointly planned by default:

```cpp
<[
    modules = expression -[gene_to_module]-> module_axis;
]>

<[
    response = modules -[module_to_gene]-> gene_axis;
]>
```

The compiler may still reuse persistent artifacts and ordinary backend optimizations, but it MUST NOT assume that it may eliminate or privately reorder the visible intermediate across the field boundary.

Putting both operations in one field exposes their transition:

```cpp
<[
    modules = expression -[gene_to_module]-> module_axis;
    response = modules -[module_to_gene]-> gene_axis;
]>
```

Field boundaries are therefore explicit optimization-visibility boundaries.

### 10.4 C++ observable behavior

A field retains C++ sequencing and observable-behavior requirements.

The compiler MUST preserve:

- data dependencies;
- volatile and atomic semantics;
- exception-visible effects;
- I/O and synchronization effects;
- effect-contract boundaries;
- reads and writes visible to ordinary C++ code;
- explicitly requested canonical or physical orders.

The compiler may execute independent operations concurrently only when their semantic and C++ effects permit it.

### 10.5 Control flow

Ordinary C++ control flow is permitted inside a field.

Compile-time control flow is analyzed normally. Runtime control flow becomes part of the semantic graph.

The compiler SHOULD avoid unbounded branch-conditioned multiversioning. It MAY:

- plan common regions around a branch;
- retain ordinary native control flow;
- produce a bounded number of profile-specialized paths;
- join alternative data states conservatively;
- place a planning boundary at an opaque or highly divergent region.

When missing branch profile information materially degrades planning, the compiler SHOULD warn rather than fail.

### 10.6 Loops

A loop inside a field retains C++ iteration semantics. The compiler may plan the loop body as a recurrent workload and use declared reuse horizons.

```cpp
<[
    given ce::persists(structure(regulation), ce::across(trajectory));
    given ce::changes(values(regulation), ce::each_iteration);
::
    for (auto step : trajectory) {
        response = expression -[regulation]-> genes;
        update_values(regulation, step);
    }
]>
```

The compiler MUST NOT assume a loop-invariant structure, order, or profile unless it proves that fact or receives an applicable contract.

### 10.7 Named fields

A named field is a C++ function with a Cellerator semantic body:

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

Named fields:

- have ordinary C++ linkage, overload, template, and access-control behavior;
- have a stable semantic identity derived from their qualified name, signature, language revision, and explicit versioning;
- may be called from ordinary C++ as compiled functions;
- may be semantically visible when called inside another field and their definition is available;
- may be separately compiled into prepared recipes when semantic inlining is not selected.

C++ `inline` retains its standard language meaning. Cellerator planning inlining is controlled separately:

```cpp
prefer ce::inline_semantics(propagate);
require ce::inline_semantics(propagate);
require ce::field_boundary(propagate);
```

A required semantic inline fails if the semantic body is unavailable or incompatible.

### 10.8 Nested fields

Revision 0.1 proposes lexical nesting with deliberately simple semantics.

A nested field:

- inherits the outer field's `given`, `prefer`, and `require` environment;
- may add facts, preferences, constraints, offers, and local inspection;
- remains visible to the outermost field's planner by default;
- does not create an execution or materialization barrier merely by being nested.

Example:

```cpp
<[
    given ce::persists(structure(regulation), ce::across(experiment));
::
    modules = expression -[gene_to_module]-> module_axis;

    <[
        prefer ce::minimum_transient_memory;
    ::
        response = modules -[module_to_gene]-> gene_axis;
    ]>
]>
```

An inner hard constraint may strengthen the inherited legal set. It cannot weaken an outer hard constraint.

A nested field becomes a real planning boundary only when it declares:

```cpp
require ce::field_boundary;
```

At such a boundary the outer planner sees the inner field's typed inputs, outputs, effects, orders, generations, and complete transition costs, but not its internal candidate graph.

This nesting model is provisional but recommended because it provides scoped policy without turning every nested field into an accidental optimization wall.

### 10.9 Field exit and asynchronous work

Execution of a field may enqueue asynchronous device work.

Field exit MUST NOT imply a hidden host synchronization. Produced states carry readiness information or participate in the caller's stream-ordering contract.

A host read, foreign call, or forced cross-provider transition that requires completion MUST use an explicit synchronization operation such as:

```cpp
ce::await(response);
```

or an effect-contracted API that declares synchronization.

### 10.10 Field identity and compiled identity

Every field has:

- a semantic source identity;
- a planning identity derived from semantic graph and applicable constraints;
- zero or more profile-specific plan identities;
- zero or more target-specific executable identities.

Anonymous-field identities may be source-location-derived and are not stable under source movement. Libraries that persist or exchange compiled artifacts SHOULD use named fields or an explicit standard-library stable identity declaration.

## 11. Planning directives and authority hierarchy

### 11.1 Directive classes

The planning prologue supports six directive classes:

```cpp
given   planning_fact;
prefer  planning_preference;
require planning_constraint;
offer   offered_object;
force   forced_object;
inspect inspection_request;
```

The expression following each directive is typed. A plain runtime `bool` is not automatically a planning fact.

### 11.2 `given`

`given` supplies a planning fact believed to describe the expected workload or data.

Examples:

```cpp
given ce::uses(regulation) >= 10000;
given ce::persists(structure(regulation), ce::across(trajectory));
given ce::changes(values(regulation), ce::each_iteration);
given ce::profileof(expression) matches activated_fibroblast;
```

A `given` fact:

- MAY affect ranking, packing, specialization, and amortization;
- MUST NOT relax semantic correctness;
- SHOULD be checked against contradictory compile-time evidence;
- SHOULD produce a warning when contradicted or materially unsupported;
- MAY be guarded at runtime only when the guard and fallback are explicit in diagnostics.

If a `given` fact is false, the program's mathematical result must remain correct. Performance, compilation cost, or selected fallback behavior may differ.

### 11.3 `prefer`

`prefer` adjusts optimization priorities without removing legal candidates:

```cpp
prefer ce::minimum_latency;
prefer ce::minimum_persistent_memory;
prefer ce::preserve_order(orderof(expression));
prefer ce::avoid_transfer;
prefer ce::measurement_over_analytical;
```

Preferences may be weighted standard-library objects:

```cpp
prefer 4 * ce::minimum_latency
     + 1 * ce::minimum_persistent_memory;
```

The exact weighting syntax is library-defined C++ operator composition.

If preferences conflict, the planner uses their declared objective semantics and reports the resulting tradeoff.

### 11.4 `require`

`require` removes candidates or plans that do not satisfy a hard condition:

```cpp
require ce::deterministic;
require ce::canonical_output(response);
require ce::persistent_bytes <= 2_GiB;
require ce::no_host_synchronization;
require ce::graph_capture_compatible;
```

If no valid complete realization satisfies all requirements, compilation or preparation MUST fail with diagnostics describing the unsatisfied constraints and nearest rejected alternatives.

A hard constraint cannot be ignored as an optimization hint.

### 11.5 `offer`

`offer` adds a programmer-supplied object to the compiler's search space:

```cpp
offer profile activated_fibroblast;
offer decomposition by_module;
offer candidate my_sparse_candidate;
offer transform fuse_regulatory_moments;
offer cost_model cluster_costs;
offer realization hand_packed_sm70;
```

An offered object:

- MUST be validated against its protocol;
- MAY be rejected for correctness or incompatibility;
- MAY lose to another legal object;
- MUST appear in candidate and cost diagnostics;
- does not become mandatory merely because the programmer supplied it.

This is the source-level form of: "Here is my design. Consider it and beat it if you can."

### 11.6 `force`

`force` selects a validated object or family and removes alternatives outside the forced scope:

```cpp
force decomposition by_module;
force candidate my_sparse_candidate;
force projection my_projection;
force realization hand_packed_sm70;
```

Forcing a decomposition still permits planning inside each resulting fragment. Forcing a candidate still permits compatible packing, projection, and launch choices unless the candidate protocol fixes them. Forcing a realization fixes the complete exposed plan.

A forced object:

- MUST pass semantic, exact-coverage, numerical, target, generation, and capability validation;
- MUST NOT bypass contribution-ownership or partial-algebra verification;
- MUST fail clearly when unavailable or incompatible;
- MUST NOT silently fall back unless the source explicitly supplies a fallback object.

The ultimate unchecked escape hatch is ordinary manual C++ or CUDA outside Cellerator planning, not an invalid Cellerator plan.

### 11.7 `inspect`

`inspect` requests compile-time records and has no semantic or ranking effect:

```cpp
inspect semantics;
inspect state_flow;
inspect candidates;
inspect costs;
inspect ir<decomposition>;
inspect ir<native>;
```

Inspection may also be requested by compiler options. Source requests are useful for durable library diagnostics and tests.

### 11.8 Precedence and conflict

The authority order is:

```text
language correctness and C++ observable behavior
    > hard requirements
    > validated forced selection
    > offered alternatives
    > preferences
    > given planning facts
    > implementation defaults
```

`force` cannot override language correctness or a hard `require`. A direct conflict between forced objects or requirements is ill-formed.

Nested fields inherit outer directives. Inner `given` facts may refine outer facts; contradictions warn. Inner requirements intersect outer requirements. Inner preferences are scoped additions. Inner force directives apply only to the enclosed semantic subgraph unless they name an outer object.

### 11.9 No performance guarantee

The compiler SHOULD pursue the best measured or modeled complete realization available. The language does not promise a globally optimal program.

Optimization reports MUST distinguish:

- analytical selection;
- empirical selection;
- cached selection;
- forced selection;
- conventional fallback;
- stale or missing evidence;
- contaminated measurements;
- practical ties within tolerance.

## 12. Representative data and data-state evolution

### 12.1 Compilation inputs

A Cellerator compilation unit may receive:

- source code;
- target and toolchain identities;
- one or more representative profiles;
- optional semantic geometry artifacts;
- optional executable or lowering-resumption artifacts;
- candidate and provider catalogs;
- measured performance evidence;
- external cost information.

Dataset paths and workflow loading are not language semantics. Build systems, compiler drivers, or standard-library tools bind profiles to source symbols and fields.

### 12.2 Abstract data state

The compiler maintains an abstract data state for compiler-semantic objects at each program point.

The state may include:

- structure identity and epoch;
- value and overlay generations;
- exact domains, axes, and orders;
- support class and active-support statistics;
- value dynamics and numeric ranges;
- profile alternatives and confidence;
- residency and readiness;
- expected reuse horizon;
- known dirty extents;
- partition and hierarchy identity.

This state is used for legality, invalidation, and planning. Not every component is known statically.

### 12.3 Profile binding

A profile may be bound externally as the default profile for a parameter, object, field, or program.

Source can refine or replace the expected profile:

```cpp
<[
    given ce::profileof(expression) matches activated_fibroblast;
::
    response = expression -[regulation]-> genes;
]>
```

A source-level profile assertion refers to a symbolic, versioned profile object. It does not contain a file path and does not make the source dependent on one data loader.

### 12.4 Automatic propagation

Every compiler-semantic operation SHOULD provide a data-state transfer function.

Examples:

- value-only arithmetic changes value statistics but preserves structure and order;
- relation transfer changes one domain axis and may change expected sparsity or distribution of the result;
- active-support update changes overlay generation and activity statistics;
- structural filtering creates a new structure identity and epoch;
- canonicalization changes order but not biological identity;
- a declared pure C++ call preserves all semantic state it does not read or write;
- an opaque mutating call widens or invalidates affected state.

The compiler SHOULD propagate profiles and bounds through such operations automatically.

### 12.5 `expect`

`expect` supplies a program-point post-state hint:

```cpp
ce::activate_fibroblasts(expression);

expect ce::profileof(expression) matches activated_fibroblast;
```

`expect`:

- has no runtime effect by default;
- may refine planning after the statement;
- must not relax correctness;
- warns when contradicted by provable state;
- is especially useful after custom or opaque transformations.

An implementation MAY support `expect` with a bounded runtime guard and explicit fallback policy, but the guard must be visible in diagnostics.

### 12.6 `verify`

`verify` requests validation:

```cpp
verify ce::profileof(expression) matches activated_fibroblast;
verify epochof(regulation) == expected_epoch;
verify orderof(expression) == persistent_gene_order;
```

The compiler decides whether a verification is compile-time, preparation-time, or runtime according to the operands. Failure is an error or runtime contract failure, not merely a warning.

A profile verification must be based on defined measurable predicates. It cannot verify a vague biological interpretation.

### 12.7 Alternative profiles

A field may describe a bounded alternative set:

```cpp
given ce::profileof(expression) in
    ce::profiles{quiescent_fibroblast, activated_fibroblast};
```

The compiler may:

- join the alternatives into one conservative profile;
- compile a bounded number of specialized plans;
- choose a runtime profile dispatch when explicitly allowed;
- retain one generic exact fallback.

The compiler SHOULD report the number of variants and the expected dispatch cost.

### 12.8 Branch joins

At a runtime control-flow join, the compiler joins the data states reaching that point.

A join preserves facts true on every path and records bounded alternatives for material differences. The compiler MUST NOT generate an exponential decision tree by default.

When profile uncertainty materially changes the likely winning realization, the compiler SHOULD emit a warning such as:

```text
cellerator: profile state after branch has 3 materially distinct alternatives;
            using joined profile for field "propagate"
            note: provide `expect profileof(expression) ...` or an explicit
                  bounded profile set to enable specialization
```

### 12.9 Runtime profile selection

Runtime profile selection is an explicit power feature. It is not the default meaning of `given`.

A future-compatible standard-library form is:

```cpp
offer ce::profile_dispatch{
    .profiles = {quiescent_fibroblast, activated_fibroblast},
    .selector = classify_activation,
    .fallback = generic_profile
};
```

The selector's cost, effects, and failure behavior participate in planning.

### 12.10 Profile correctness boundary

A profile may change:

- which legal candidate wins;
- whether packing is amortized;
- how many specializations are emitted;
- which performance evidence is considered fresh;
- whether a warning is produced.

A profile MUST NOT change the exact mathematical meaning of a relation, the identity of an edge, the required output effect, or whether an approximate cover is treated as exact.


## 13. Persistence, reuse, identity, and biological time

### 13.1 Independent lifetime layers

Cellerator distinguishes at least these lifetime layers:

1. domain and canonical biological identity;
2. immutable relation structure;
3. structure epoch;
4. persistent order and recovery maps;
5. projection and packed structure;
6. relation values;
7. active-support overlay;
8. mutable dense state;
9. prepared executable recipe;
10. launch-time pointers, streams, readiness, and workspace;
11. performance evidence and preference freshness.

These layers MUST NOT share one undifferentiated "version."

### 13.2 Structure epochs

A structure epoch identifies one exact structural generation of a persistent relation identity.

The following normally advance the structure epoch:

- adding or removing logical edges;
- changing stable logical edge identity;
- changing source or destination axis identity;
- changing logical edge order without an explicit compatible transform;
- changing exact segmentation or structural hierarchy required by the relation;
- replacing structure under the same persistent identity.

A structure-epoch change invalidates dependent value maps, projections, packed operands, prepared contracts, and executable recipes according to their declared dependencies.

### 13.3 Value generations

A value generation identifies one published numerical generation bound to a compatible structure epoch.

Changing relation weights, mutable state, gradients, or another value plane advances the applicable value generation.

A value-generation change does not inherently invalidate:

- immutable structure;
- semantic geometry based only on structure;
- structure-only decomposition;
- compatible projections;
- a prepared executable whose value binding is dynamic.

It may invalidate packed values or graph-capture bindings that depend on the old generation.

### 13.4 Active-support generations

An active-support overlay has its own generation. It is tied to a relation structure and logical edge order.

Changing active bits advances the overlay generation. It does not remove the underlying logical edges and does not advance the relation's structure epoch.

This distinction lets Cellerator reuse stable topology while activity changes rapidly.

### 13.5 Persistent order

A persistent order may outlive many value generations and operations.

A producer may leave output in persistent physical order when:

- the output contract permits packed order;
- every visible consumer can use that order or has an explicit transform;
- any ordinary C++ observer sees a compatible view or an explicit canonicalization;
- the order identity remains generation-compatible.

Canonical order is not the universal internal normal form.

### 13.6 Planning facts for reuse

Reuse facts are expressed through standard-library planning objects:

```cpp
given ce::persists(structure(regulation), ce::across(trajectory));
given ce::persists(orderof(expression), ce::across(experiment));
given ce::changes(values(regulation), ce::each_iteration);
given ce::support_evolves(active_edges, ce::slowly);
given ce::uses(regulation) >= 10000;
```

The standard library may provide biological interval objects such as `trajectory`, `cell_cycle`, `experiment`, `batch`, or user-defined epochs. They are ordinary typed values, not hard-coded language concepts.

A reuse fact affects cost amortization and artifact eligibility. It does not grant permission to use stale structure or values.

### 13.7 Epoch boundaries

A source program may explicitly mark a semantic epoch transition:

```cpp
ce::end_epoch(regulation, differentiation_stage);
```

The exact constructor is library-defined. Its effect contract states which identities or generations advance.

For ordinary code, the compiler and standard-library owners SHOULD advance generations automatically from recognized mutation.

### 13.8 Expert identity control

The intrinsic queries are:

```cpp
identityof(object)
epochof(structure_or_relation)
generationof(value_or_state_or_overlay)
structure(object)
values(object)
supportof(object)
orderof(object)
profileof(object)
```

Experts may use versioned library operations such as:

```cpp
ce::publish_generation(weights, next_generation);
ce::advance_epoch(topology, next_epoch);
ce::rebind_identity(view, persistent_identity);
```

These operations MUST validate monotonicity and dependency closure where applicable. Unchecked identity fabrication requires an explicitly unsafe API.

### 13.9 Readiness and publication

A produced generation is not usable merely because a launch was enqueued.

Cellerator associates published generations with readiness state. A consumer on a compatible stream or provider may rely on explicit stream ordering. Cross-stream or cross-provider use requires a readiness event, lease, or equivalent provider-neutral contract.

Publication MUST occur only after successful enqueue or completion according to the provider contract. A failed operation MUST NOT publish its promised generation.

### 13.10 Artifact reuse and lowering resumption

A compiled artifact may resume lowering at a compatible stage:

```text
canonical source
atom evidence
semantic atom
target cover
physical projection
packed operand
executable recipe
local realization
```

The language does not expose these as implicit caches. Compiler drivers and standard-library artifact APIs may offer compatible artifacts to a field.

Invalidation follows the earliest incompatible dependency:

- stale value generation resumes no later than packed operand;
- target or toolchain mismatch resumes no later than physical projection;
- order mismatch resumes no later than semantic atom;
- structure-epoch mismatch resumes no later than atom evidence;
- corrupt or identity-mismatched artifacts resume from canonical source.

The exact current internal stage names are not all language ABI. The dependency principle is language-defined.

## 14. Numerical, determinism, output, and order contracts

### 14.1 Numerical policy is semantic

Cellerator operations may distinguish:

- relation-value storage type;
- state storage type;
- multiplication type;
- accumulation type;
- output storage type;
- scalar type;
- rounding policy;
- saturation policy;
- NaN policy;
- infinity policy;
- quantization policy.

A numerical policy may be attached to a relation, state, operation, field, candidate, decomposition, or partial algebra through typed standard-library objects.

Example:

```cpp
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
```

The compiler MUST NOT silently select a numerically weaker policy than required.

### 14.2 Determinism

Determinism is a hard semantic or reproducibility constraint when required:

```cpp
require ce::deterministic;
require ce::fixed_reduction_tree;
require ce::stable_work_order;
```

A candidate using nondeterministic atomics is illegal when the applicable contract forbids them.

A partial-result algebra that requires a deterministic tree MUST name or derive a compatible tree identity.

A soft preference may instead say:

```cpp
prefer ce::deterministic_when_practical;
```

That preference does not promise bitwise stability.

### 14.3 Nonfinite behavior

Segmented reductions and normalizations MUST define empty and singleton behavior. Numerical policies MUST define whether NaN and infinity values propagate, reject, or saturate.

The compiler cannot infer a nonfinite policy from a backend library's defaults.

### 14.4 Output effects

The language-defined output effects are:

- overwrite;
- accumulate;
- affine accumulate;
- partial write.

Assignment syntax expresses the common cases. Standard-library lvalue adapters express affine and partial effects.

A partial write MUST identify exact output coverage and contribution ownership. If several partial writers overlap, exact reconstruction requires a verified partial algebra.

### 14.5 Order requirements

An operation may require:

- preserve the current persistent order;
- allow a packed or projection-native order;
- produce canonical order.

Examples:

```cpp
prefer ce::packed_output(response);
require ce::canonical_output(response);
require ce::preserve_order(response);
```

An explicit order transform is a semantic operation:

```cpp
canonical_response = ce::canonicalize(response);
reordered = ce::reorder(response, module_major_order);
```

The planner MUST include bytes moved, workspace, synchronization, and downstream effects in the complete cost.

### 14.6 Aliasing

Input-output aliasing is illegal unless both the operation contract and selected candidate permit it.

A C++ type being assignable or two pointers being equal is insufficient evidence.

An aliasing requirement can be stated:

```cpp
require ce::in_place(response);
```

Compilation fails if no legal in-place realization exists.

### 14.7 Gradients and transpose closure

Gradient-producing code uses explicit semantic operations and the same relation identities:

```cpp
source_gradient +=
    destination_gradient -[transpose(regulation)]-> source_genes;

value_gradient = ce::contract_on(
    supportof(regulation),
    source_state,
    destination_gradient,
    ce::multiply);
```

A field may require forward, transpose, source-gradient, destination-gradient, or value-gradient closure. These requirements participate in projection and candidate selection.

Cellerator does not assign model, loss, optimizer, or causal meaning to these operations.

## 15. C, C++, CUDA, and native interoperability

### 15.1 Outside an execution field

Outside a field, ordinary C++ and CUDA retain their host-language behavior. Cellerator compiler-semantic types may be passed through ordinary functions, but no cross-statement Cellerator planning occurs unless a named field is called or another field is opened.

### 15.2 Unknown calls inside a field

An unknown call inside a field is an opaque semantic node:

```cpp
<[
    modules = expression -[gene_to_module]-> module_axis;
    hand_tuned_kernel(modules);
    response = modules -[module_to_gene]-> gene_axis;
]>
```

Unless the compiler has an effect contract, it MUST conservatively assume that the call may:

- read and write reachable memory;
- change value generations;
- invalidate data-state profiles;
- synchronize or transfer;
- observe order and residency;
- throw or perform other C++-visible effects.

The planner may split the field around the call and materialize a compatible boundary. It MUST report the barrier when it prevents optimization.

### 15.3 Effect contracts

A function, method, lambda, or callable object may carry a trailing Cellerator effect contract:

```cpp
void update_regulatory_values(
    relation_values<float, gene, gene>& weights,
    state<float, cell, gene> expression)
effects(
    reads(expression),
    mutates(weights),
    advances(generationof(weights)),
    preserves(structure(weights), orderof(weights)),
    deterministic
);
```

The proposed effect vocabulary includes:

```text
reads(...)
writes(...)
mutates(...)
preserves(...)
invalidates(...)
advances(...)
publishes(...)
canonicalizes(...)
reorders(...)
transfers(...)
allocates(...)
synchronizes(...)
aliases(...)
deterministic
pure
opaque
```

Effect arguments are compiler-semantic objects or layers such as structure, values, support, order, profile, generation, residency, or readiness.

### 15.4 Effect meanings

`reads(x)` declares a read dependency.

`writes(x)` declares output but need not imply that prior contents are read.

`mutates(x)` declares a read-write dependency.

`preserves(layer...)` states that listed semantic layers remain compatible.

`invalidates(layer...)` discards applicable abstract state and artifacts.

`advances(generationof(x))` declares publication of a later generation.

`publishes(x)` declares that the resulting generation becomes visible through a readiness contract.

`canonicalizes(x)` and `reorders(x)` state order effects.

`transfers(x)` declares residency movement.

`allocates` and `synchronizes` expose effects that must enter planning.

`deterministic` declares deterministic behavior under the stated numeric contract.

`pure` means no externally observable effect except its return value and reads of explicitly declared immutable inputs.

`opaque` requests an intentional barrier and suppresses missing-contract warnings for that call.

### 15.5 Verification of effects

When a function body is available, the compiler SHOULD verify its declared effects against visible operations.

For an external or separately compiled function, the effect contract is part of its ABI and may be trusted subject to optional runtime validation.

A contradiction between visible behavior and a declared effect is an error.

Omitting an effect that the compiler cannot prove produces a conservative barrier, not undefined behavior.

### 15.6 CUDA kernels

A CUDA kernel may be called manually inside or outside a field. Inside a field it should have an effect contract:

```cpp
void launch_custom_gate(
    relation<float, gene, gene> regulation,
    active_support<gene, gene>& active,
    cudaStream_t stream)
effects(
    reads(regulation),
    mutates(active),
    advances(generationof(active)),
    preserves(structure(regulation), orderof(regulation)),
    deterministic
);
```

The kernel launch geometry remains CUDA syntax and programmer-controlled. Cellerator does not reinterpret `<<<...>>>`.

A manual CUDA call may also be wrapped as a custom candidate so that it competes in Cellerator planning.

### 15.7 Native fallback

When a statement has no Cellerator semantic operation and no applicable planner protocol, it is compiled normally by the host compiler.

A field may contain ordinary arithmetic, pointer code, templates, intrinsics, library calls, and inline assembly. The compiler treats them according to ordinary C++ semantics plus any declared effects.

Cellerator MUST remain humble at this boundary: a manual implementation is legitimate, not an error condition.

### 15.8 Exceptions and termination

A C++ exception crossing asynchronous device work is not implicitly synchronized.

An effect-contracted call that may throw remains an ordering boundary unless the compiler proves the exception cannot observe unfinished work. Device-side failure follows the selected backend's explicit error contract.

The precise interaction of C++ exceptions with field-wide speculative planning remains an open design area. Revision 0.1 requires preservation of observable exception order and forbids hidden synchronization used only to simplify implementation.

## 16. Exact coverage, decomposition, atoms, and extents

### 16.1 Expert status

Exact coverage, decomposition, atoms, extents, projections, and partial results are first-class expert programming concepts.

They are not required for ordinary Cellerator code. They are exposed because they determine correctness and performance in the current compiler, and because expert programmers must be able to contribute better structures than the automatic search.

### 16.2 Exact coverage

A `coverage` object names an exact subset of logical work associated with:

- persistent coverage identity;
- structure identity and epoch;
- source and destination axes;
- exact logical count;
- membership representation;
- coverage roles.

Built-in membership forms may include intervals, explicit entity IDs, logical edge IDs, semantic components, segment sets, and unions. Providers may define versioned additional forms.

Shape equality does not establish coverage equality.

### 16.3 Coverage roles

A coverage may participate as:

- certified exact coverage;
- approximate proposal membership;
- exact read requirement;
- read-only halo;
- physical replica;
- exclusive output owner;
- partial contribution owner.

Proposal membership is not executable exactness. A read halo is not an output contributor. A replica does not acquire biological identity independent of its source.

Every exact logical contribution has one exclusive owner unless a declared partial-result algebra proves reconstruction.

### 16.4 Decomposition

A `decomposition` is a typed set of legal alternatives for splitting one semantic operation.

The standard library should offer a fluent builder:

```cpp
decomposition by_module =
    ce::decompose(regulation)
        .split(ce::semantic_components(modules))
        .read_halo(module_halo)
        .output(ce::partial(response, partial_sum))
        .fallback(ce::unsplit);
```

Supported split dimensions may include:

- source axis;
- destination axis;
- relation edges;
- semantic components;
- segments;
- modules;
- physical extents.

A decomposition alternative states its exact input and output coverages, orders, replication, halos, partial algebra, numerical policy, and candidate family.

A portfolio MUST contain a complete exact fallback unless the enclosing hard constraint intentionally makes failure preferable.

### 16.5 Partial-result algebra

A `partial_algebra` proves how overlapping or disjoint partial results reconstruct the exact result.

Example:

```cpp
partial_algebra partial_sum =
    ce::partial_algebra<float>("gene-response-sum")
        .neutral(0.0f)
        .merge(ce::add)
        .finalize(ce::identity)
        .associative()
        .commutative()
        .numeric(ce::f32_accumulation);
```

A partial algebra may declare:

- associativity;
- commutativity;
- idempotence;
- ordered-only merging;
- a required deterministic tree;
- state layout and alignment;
- neutral, merge, and finalize operations;
- numerical policy.

Flags are proof obligations, not wishes. The compiler SHOULD verify built-in algebras and MAY require tests, proofs, or explicit trust for external algebras.

### 16.6 Atoms

An `atom` is an independently nameable unit at a declared compiler level.

There is no universal smallest atom. An atom must state enough to be:

- exactly related to logical coverage;
- independently bound or materialized where applicable;
- invalidated through explicit dependencies;
- consumed or produced by at least one operation;
- composed through a declared algebra or dependency.

An atom may have planes for immutable structure, mutable relation values, active support, mutable state, gradients, partial results, dense results, physical views, readiness, and leases.

The language-level abstraction MUST NOT equate an atom with a disk shard, one allocation, one kernel block, or one biological module.

### 16.7 Atom requirements and affordances

A candidate may request atom properties:

- accepted atom species;
- exact coverage;
- required planes;
- storage, logical, and accumulation numeric types;
- local index width;
- order;
- alignment and contiguity;
- mutability and generation policy;
- graph-stable addresses;
- number of extents;
- permitted transform paths.

An available atom advertises corresponding affordances:

- exact coverage;
- physical encoding;
- projection ABI;
- planes and generations;
- multi-extent legality;
- gradient and output support;
- persistence eligibility;
- graph-stable address;
- fused transforms.

Binding succeeds only when requirements and affordances are compatible.

### 16.8 Extents

An `extent` is one physical span contributing to an atom or port binding. It includes explicit:

- atom identity;
- address;
- byte and element count;
- stride;
- alignment;
- value generation;
- residency.

A logical port may bind several extents. A candidate may consume them directly or require explicit assembly.

Assembly is a planner-visible operation with bytes, workspace, launch, and synchronization cost. It is never a hidden convenience.

### 16.9 Hierarchical index spaces

A relation may exceed one kernel-local index range or consist of independently bounded components.

A hierarchical index space preserves:

- global 64-bit extent;
- stable component identity;
- aggregate logical offset;
- explicit local index width;
- local-to-global recovery;
- optional global identity sidecars.

The aggregate need not be physically contiguous.

### 16.10 Physical padding

Physical padding, empty matrix-engine slots, alignment gaps, transport framing, and unused extent capacity have no biological identity and are not logical work items.

A physical candidate MUST map each real contribution to exact logical identity and ownership.

## 17. Custom candidates, cost models, and forced realization

### 17.1 Candidate protocol

A `candidate` is a source-linked implementation alternative for one or more semantic operations.

A candidate declaration or builder must provide or resolve:

- stable candidate and provider identities;
- supported operation identities;
- source and destination type constraints;
- numerical and determinism capabilities;
- exact coverage and decomposition compatibility;
- required projection or view ABI;
- required atom affordances;
- target capabilities;
- resource query;
- preparation function;
- launch function or executable-stage builder;
- output and effect contracts;
- analytical cost;
- measurement hook or evidence policy;
- graph-capture and persistence properties.

Illustrative library syntax:

```cpp
candidate my_relation_candidate =
    ce::candidate("lab.regulation.sm70.hybrid.v1")
        .supports(ce::relation_apply)
        .requires(ce::target_capability("nvidia.sm70"))
        .accepts(ce::f16_multiply_f32_accumulate)
        .projection(my_projection_abi)
        .atoms(my_atom_requirements)
        .resources(query_resources)
        .prepare(prepare_candidate)
        .launch(launch_candidate)
        .effects(candidate_effects)
        .cost(analytical_cost)
        .measure(measure_candidate);
```

This is ordinary Cellerator library code building a compiler-semantic object. The base language does not encode every field as a keyword.

### 17.2 Source-linked registration

Candidate and provider discovery SHOULD occur at compile or preparation time from source-linked, immutable catalog fragments.

Prepared hot paths MUST NOT scan dynamic plugin directories or perform unbounded discovery.

A library may publish several candidate fragments and architecture-specific variants.

### 17.3 Candidate competition

An offered candidate enters the same legal and cost analysis as built-in candidates:

```cpp
<[
    offer candidate my_relation_candidate;
    inspect candidates, costs;
::
    response = expression -[regulation]-> genes;
]>
```

The report must say whether the candidate was:

- malformed;
- semantically incompatible;
- numerically incompatible;
- missing a capability;
- rejected by memory or determinism constraints;
- analytically dominated;
- empirically slower;
- contaminated during measurement;
- tied within tolerance;
- selected.

### 17.4 Custom cost models

A `cost_model` may assign or reprice complete costs such as:

- fixed overhead;
- persistent and transient bytes;
- transfer bytes;
- communication bytes;
- launch and synchronization;
- expected reuse credit;
- acquisition and assembly.

```cpp
offer cost_model cluster_costs;
```

A cost model MAY change ranking and frontier construction. It MUST NOT declare an incorrect candidate correct or waive an exact constraint.

External cost exchange may iterate through a bounded set of proposals. The compiler remains responsible for local semantic validation.

### 17.5 Forced candidate and forced realization

Forcing a candidate fixes the implementation family:

```cpp
force candidate my_relation_candidate;
```

Forcing a realization fixes the exposed decomposition, cover, projection, packing, stage graph, and candidate assignments:

```cpp
force realization my_sm70_realization;
```

A realization may be target-specific and toolchain-specific. Diagnostics MUST state which aspects are portable and which are locked.

### 17.6 Manual implementation

A programmer may bypass Cellerator planning:

```cpp
manual_regulation_kernel<<<grid, block, shared, stream>>>(
    expression_ptr, relation_ptr, response_ptr);
```

This is ordinary CUDA. Cellerator does not claim ownership of the decision.

To remain visible inside a larger field, the call may carry an effect contract or be wrapped as a candidate. Otherwise it is an opaque barrier.

### 17.7 Conventional fallback

A semantic operation SHOULD have at least one exact conventional fallback when practical.

If no Cellerator-specific candidate applies, a library-backed or manually supplied native implementation may remain legal. Cellerator should fail only when the semantics or hard constraints cannot be satisfied, not merely because an adaptive optimization was unavailable.


## 18. Intermediate representation as a programming feature

### 18.1 Requirement

Cellerator IR is not solely an internal debugging artifact. The language MUST provide typed, versioned access to compiler representations at meaningful abstraction levels.

Ordinary users do not need IR. Expert users must be able to inspect what the compiler understood, add alternatives, transform eligible representations, and intercept lowering without writing textual compiler assembly.

### 18.2 Core IR levels

Revision 0.1 proposes these public core levels:

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

Their meanings are:

**`ir<semantic>`**
Typed domains, axes, relations, operations, effects, output contracts, control dependencies, data-state transitions, and field boundaries.

**`ir<geometry>`**
Target-independent semantic geometry and workload evidence relevant to candidate formation, including exact structure plus profile-derived statistics and provenance.

**`ir<decomposition>`**
Exact split alternatives, coverages, ownership, halos, replicas, partial algebras, order requirements, and fallback closure.

**`ir<cover>`**
One or more exact selected semantic or target covers, atom bindings, contribution ownership, and recovery maps.

**`ir<projection>`**
Target-specific physical views and projection contracts, still independent of live launch pointers.

**`ir<packed>`**
Prepared packed operands, value-position maps, generation dependencies, and explicit assembly or canonicalization work.

**`ir<executable>`**
Prepared stages, candidate assignments, dependencies, resource requirements, launch bindings, and resumable executable recipes.

**`ir<native>`**
Backend IR or generated C++, CUDA, LLVM IR, PTX, object code, or another target-specific representation. Its exact form is implementation-defined.

### 18.3 Privileged extension levels

A privileged compiler component may add versioned IR levels such as:

```text
evidence
atom
composition
basis
global_schedule
topology
```

These levels are not required for standalone Cellerator. Their ownership and stability must be explicit. In the current architecture, global discovery, persistence, placement, and transport belong above independent libCellerator rather than silently becoming core language semantics.

### 18.4 IR queries

A field or named field may request human-readable IR:

```cpp
inspect ir<semantic>;
inspect ir<decomposition>;
inspect ir<native>;
```

Programmatic compile-time access uses:

```cpp
consteval ir<semantic> semantic_graph =
    ir_of<semantic>(propagate);

consteval ir<decomposition> alternatives =
    ir_of<decomposition>(propagate, activated_fibroblast);
```

`ir_of` is a compiler intrinsic. Its result is a typed immutable view by default.

A query may name a field, operation, source range, profile, target, candidate, or compiled identity.

### 18.5 IR builders and transforms

Writable IR uses typed builders and verified transactions:

```cpp
transform ir<semantic> fuse_regulatory_moments(
    ir<semantic> graph)
{
    auto edit = ce::ir::rewrite(graph);

    edit.match(ce::ir::two_moments_of_same_relation())
        .replace_with(ce::ir::paired_relation_moments());

    return edit.commit();
}
```

A transform function:

- executes at compile or preparation time;
- has no ordinary runtime side effects;
- accepts and returns one declared IR level;
- must preserve stable identities or explicitly map them;
- must satisfy the verifier for its level;
- must declare any additional proof, target, profile, or measurement requirements.

Transforms are offered or forced through the field prologue:

```cpp
offer transform fuse_regulatory_moments;
```

An offered transform creates an alternative. It does not silently rewrite the only semantic program.

### 18.6 Semantic preservation

A semantic IR transform MUST prove or verify semantic equivalence under the applicable contract.

A decomposition or lower-level transform may alter physical organization but MUST preserve:

- exact logical coverage;
- contribution ownership;
- output effects;
- required order boundaries;
- numerical and determinism constraints;
- lifetime dependencies;
- C++ observable effects.

If equivalence cannot be established, the transform may instead publish a custom semantic operation or candidate with an explicit contract.

### 18.7 Lowering interception

An expert may intercept a named lowering boundary:

```cpp
offer ce::lowering<projection>(
    my_projection_lowering);

force ce::lowering<packed>(
    my_packed_operand_builder);
```

A lowering hook receives typed input IR and caller-owned capacity or builder interfaces. It returns validated output IR plus declared costs and provenance.

Lowering interception MUST NOT bypass the verifier for the destination level.

### 18.8 IR inlining

Named field bodies and semantic operation definitions may be made visible to a caller's semantic graph.

```cpp
prefer ce::inline_semantics(propagate);
require ce::inline_semantics(propagate);
```

Semantic inlining means replacing a field-call node with its semantic IR for joint planning. It is distinct from native machine-code inlining.

After Cellerator lowering, ordinary Clang/GCC inlining, LTO, and CUDA device inlining remain available.

### 18.9 Stability guarantees

The proposed stability classes are:

- semantic IR: stable within one Cellerator language major version;
- geometry, decomposition, and coverage IR: public, versioned schema;
- projection, packed, and executable IR: public, target-aware, versioned schema;
- native IR: implementation-defined and toolchain-specific;
- human-readable dumps: diagnostic format, not a stable parser API.

A compiler MUST reject incompatible writable IR rather than reinterpret it.

### 18.10 IR provenance

Every exposed IR object SHOULD carry or resolve:

- source field and operation identity;
- language and IR schema version;
- profile and evidence identity;
- structure epochs and value-generation dependencies;
- target, runtime, driver, library, and toolchain identities where relevant;
- transform history;
- validation status;
- cost and measurement provenance.

This provenance lets a programmer understand not only what IR exists, but why it exists and when it may be reused.

## 19. Diagnostics and introspection

### 19.1 Optimization records are part of the product

A conforming implementation MUST provide an optimization record for each planned field on request.

The record SHOULD be machine-readable and human-readable, with stable identifiers connecting source operations, IR nodes, candidates, costs, and emitted stages.

### 19.2 Required questions

The diagnostic system must be able to answer:

- What domains, axes, relations, supports, and effects were inferred?
- Which source operation produced each semantic IR node?
- Which structure epochs and value generations are required?
- Which profile or joined profile applies at each operation?
- What state was propagated or invalidated?
- Which candidates and decompositions were legal?
- Why was each rejected candidate rejected?
- Which complete plan won, from which selection source, and with what confidence?
- What preparation, packing, transfer, communication, canonicalization, synchronization, and memory costs were counted?
- What was fused, split, replicated, assembled, or left conventional?
- Which persistent artifacts were reused?
- Which hints were ignored or contradicted?
- Where did an opaque C++ call stop reasoning?
- Which IR levels were emitted?
- What reached the host and device backends?

### 19.3 Source requests

Examples:

```cpp
inspect semantics, state_flow;
inspect candidates, costs;
inspect persistence;
inspect barriers;
inspect ir<projection>, ir<native>;
```

The compiler driver should additionally support global flags for filtering by field, operation, source range, candidate, target, and diagnostic category.

### 19.4 Counterfactual reports

The compiler SHOULD support bounded counterfactual planning:

```cpp
inspect ce::compare(
    ce::current_plan,
    ce::without(ce::persists(structure(regulation))),
    ce::with(ce::uses(regulation) == 1));
```

A counterfactual report must not mutate the selected program unless separately requested. It is a planning query.

### 19.5 Explainability limits

Diagnostics must distinguish facts from estimates.

For example:

```text
exact: candidate requires 64 MiB persistent projection
measured: median kernel time 83.4 us, 15 samples
modeled: order transform 12.1 us
given: structure reuse 10,000
inferred: active support density 0.18 to 0.24
unknown: cross-node transfer price
```

The compiler must not present an analytical estimate as measured evidence.

## 20. Errors, warnings, and fallback

### 20.1 Errors

Compilation or preparation MUST fail for:

- statically incompatible biological domains;
- incompatible runtime identities required by a prepared contract;
- stale hard structure epochs or value generations;
- invalid output-update or aliasing semantics;
- invalid exact coverage;
- duplicate or missing exact contribution ownership;
- an invalid partial-result algebra;
- an unsatisfied hard constraint;
- an unavailable forced object without explicit fallback;
- contradictory effect contracts;
- incompatible IR schema;
- a target-specific forced plan on an incompatible target;
- silent numerical-policy weakening.

### 20.2 Warnings

The compiler SHOULD warn for:

- a materially important missing profile hint;
- a profile contradiction;
- branch-state widening that changes likely plan choice;
- an ignored or dominated preference;
- an opaque call that blocks planning;
- an implicit field split caused by C++ effects;
- a costly canonicalization or order boundary;
- stale performance evidence;
- analytical fallback after failed measurement;
- a target-locked forced plan;
- a one-shot workload paying substantial preparation cost;
- an offered object that is legal but never competitive;
- an identity or generation manually controlled without a verification;
- a compiler-semantic object used only through an opaque ordinary-code path where useful reasoning is lost.

Warning groups MUST be individually controllable. A source-level `expect` should not become a correctness assertion merely because warnings are promoted to errors.

### 20.3 Conventional fallback

When automatic data-aware planning cannot improve an operation, the compiler may choose a conventional exact implementation.

A conventional fallback is not a semantic downgrade. Diagnostics should state that no adaptive candidate was selected and why.

### 20.4 No silent recovery from force

A failed `force` directive is an error unless the source explicitly names fallback behavior:

```cpp
force ce::first_available{
    hand_packed_sm70,
    conventional_sparse
};
```

The fallback order is part of the program and must appear in diagnostics.

## 21. Compilation model

### 21.1 Conceptual pipeline

A conforming compiler behaves as if it performs these phases:

```text
C++ preprocessing and parsing
Cellerator syntax and type analysis
effect and lifetime analysis
semantic field IR construction
representative-data state propagation
semantic geometry and exact coverage formation
decomposition and candidate discovery
candidate catalog and capability filtering
connected-operation and transition planning
analytical and optional empirical selection
projection, packing, and executable preparation
native host/device lowering
ordinary backend optimization and code generation
```

An implementation may combine phases so long as observable semantics, diagnostics, and exposed IR remain consistent.

### 21.2 Semantic and physical inputs

A Cellerator compilation may be purely ahead-of-time, profile-guided, preparation-time, or JIT-assisted.

The same source semantics must be preserved whether geometry is:

- compiled now;
- loaded from a semantic artifact;
- loaded from a compatible target-specific execution artifact;
- adapted from a compatibility artifact;
- supplied by an external exact provider.

The acquisition route is a planning input, not a different language meaning.

### 21.3 Persistent planning keys

A reusable plan identity may depend on:

- mathematical problem identity;
- persistent structure identities and epochs;
- semantic geometry;
- target performance class;
- runtime, driver, library, and kernel build;
- numerical, determinism, output-order, and graph policies;
- partition hierarchy;
- reuse assumptions;
- profile and evidence revision.

Live pointers and mutable launch bindings are not persistent semantic keys.

### 21.4 Complete cost

Candidate selection SHOULD account for complete amortized cost, including:

- host preparation;
- semantic packing;
- projection construction;
- backend preparation;
- static value packing;
- host-to-device transfer;
- dynamic input packing;
- kernel execution;
- epilogue;
- order transformation;
- synchronization;
- communication;
- device-to-host transfer;
- persistent and transient memory;
- acquisition and assembly;
- expected reuse.

Kernel time alone is not a sufficient default objective.

### 21.5 Prepared execution

A planned field may lower to a prepared stage graph containing stable stage and candidate identities, explicit dependencies, launch bindings, workspace requirements, and prepared state.

Changing pointers or a compatible dynamic value generation should not require rebuilding immutable structure-only stages.

Execution validates applicable epochs, generations, axes, orders, residency, readiness, aliasing, and workspace before dispatch.

### 21.6 Backend handoff

After Cellerator has lowered semantic operations to native operations and generated code, ordinary Clang, GCC, LLVM, NVCC, or another backend may optimize the result.

Cellerator MUST preserve sufficient source and IR mapping for diagnostics through this handoff where the backend supports it.

## 22. Standard-library boundary

### 22.1 Base-language responsibilities

The base language should remain small. Revision 0.1 assigns these responsibilities to language semantics:

- source-file opt-in;
- nominal biological domain declaration;
- compiler-semantic type protocol;
- execution fields and named fields;
- relation transfer syntax;
- planning-directive hierarchy;
- program-point `expect` and `verify`;
- effect contracts;
- intrinsic semantic queries;
- typed IR reflection and transform declaration;
- rules for identities, epochs, generations, output effects, exactness, and planning visibility.

### 22.2 Standard-library responsibilities

The Cellerator standard library should provide:

- common biological domain tags;
- owners, allocators, buffers, views, and binders;
- relation and state construction;
- file or framework adapters;
- profiles and workload-fact constructors;
- semantic operation intrinsics other than relation transfer;
- order, support, segmentation, and hierarchy builders;
- persistence and epoch helpers;
- numerical and determinism policy objects;
- decomposition, coverage, atom, extent, projection, candidate, and realization builders;
- standard partial algebras;
- cost models and measurement adapters;
- IR builders and rewrite utilities;
- artifact loading, validation, and resumption;
- C++, CUDA, and framework interoperability;
- conventional exact fallback implementations.

These constructions may be header-provided, module-provided, source-included, or generated. They are not assumed to be opaque precompiled bioinformatics routines.

### 22.3 Compiler recognition

A library type or function participates in Cellerator semantics through a versioned compiler protocol, not by spelling alone.

Users may implement compatible types and operations without placing them in the `cellerator` namespace, subject to protocol conformance and stable identity requirements.

### 22.4 Convenience levels

The library may offer multiple levels of convenience:

```cpp
ce::state_buffer<float, cell, gene>
ce::relation_storage<float, gene, gene>
ce::bio::gene_axis
ce::bio::regulatory_relation
```

These conveniences must lower to the same compiler-semantic contracts available to low-level code. They must not form a separate, less capable language.

## 23. Compact grammar sketch

This grammar is informative and intentionally incomplete with respect to ordinary C++.

```ebnf
cellerator-pragma
    ::= "#pragma" "cellerator" [ revision ]

domain-declaration
    ::= "domain" identifier ";"

execution-field
    ::= "<[" field-content "]>"

field-content
    ::= statement-seq
     | planning-directive-seq "::" statement-seq

planning-directive
    ::= "given" planning-expression ";"
     |  "prefer" planning-expression ";"
     |  "require" planning-expression ";"
     |  "offer" [ offered-kind ] expression ";"
     |  "force" [ forced-kind ] expression ";"
     |  "inspect" inspection-list ";"

named-field-definition
    ::= "field" function-declarator execution-field

relation-transfer-expression
    ::= assignment-expression
        "-[" relation-selector "]->"
        destination-axis-expression

relation-selector
    ::= expression
        [ "on" axis-expression ]
        [ "where" support-expression ]

program-point-directive
    ::= "expect" planning-expression ";"
     |  "verify" verification-expression ";"

effect-specifier
    ::= "effects" "(" effect-list ")"

ir-type
    ::= "ir" "<" ir-level ">"

transform-definition
    ::= "transform" ir-type function-declarator compound-statement
```

The final grammar should minimize conflict with C++ template tokens, lambda captures, attributes, and operators. The lexer recognizes `<[` and `]>` only after the Cellerator pragma is active.

## 24. Integrated examples

### 24.1 Ordinary semantic program

```cpp
#pragma cellerator 0.1

#include <cellerator/cellerator.hh>

namespace cardiac {

domain cell;
domain gene;

field void regulatory_response(
    state<float, cell, gene> expression,
    relation<float, gene, gene> regulation,
    axis<gene> target_genes,
    state<float, cell, gene>& response)
<[
    response = expression -[regulation]-> target_genes;
]>

} // namespace cardiac
```

### 24.2 Data-aware persistent trajectory

```cpp
field void follow_activation(
    state<float, cell, gene>& expression,
    relation<float, gene, gene>& regulation,
    axis<gene> genes,
    ce::trajectory steps)
<[
    given ce::persists(structure(regulation), ce::across(steps));
    given ce::persists(orderof(expression), ce::across(steps));
    given ce::changes(values(regulation), ce::every(steps.step()));
    given ce::uses(regulation) == steps.size();
    prefer ce::minimum_amortized_latency;
    inspect persistence, costs;
::
    for (auto step : steps) {
        expression = expression -[regulation]-> genes;
        update_regulatory_values(regulation, step);
    }
]>
```

`update_regulatory_values` must have an effect contract if the compiler is to preserve structure and order reasoning through the call.

### 24.3 Explicit active support

```cpp
<[
    given ce::persists(structure(regulation), ce::across(experiment));
    given ce::support_evolves(active_edges, ce::every(batch));
::
    ce::update_active_support(active_edges, receptor_predicate);

    response =
        expression
        -[regulation where active_edges]->
        target_genes;
]>
```

The overlay generation changes. The structure epoch does not.

### 24.4 Offered decomposition and candidate

```cpp
<[
    offer decomposition by_regulatory_module;
    offer candidate lab_sm70_hybrid;
    prefer ce::minimum_latency;
    require ce::deterministic;
    inspect candidates, costs, ir<decomposition>;
::
    response = expression -[regulation]-> target_genes;
]>
```

Both offered objects may lose. If the programmer wants to mandate them:

```cpp
force decomposition by_regulatory_module;
force candidate lab_sm70_hybrid;
```

### 24.5 Opaque and contracted native calls

```cpp
void mutate_weights(
    relation_values<float, gene, gene>& weights,
    const float* delta)
effects(
    reads(delta),
    mutates(weights),
    advances(generationof(weights)),
    preserves(structure(weights), orderof(weights)),
    deterministic
);

<[
    response = expression -[regulation]-> genes;
    mutate_weights(values(regulation), delta);
    next = response -[regulation]-> genes;
]>
```

The effect contract allows Cellerator to reuse structure and projection preparation across the call while respecting the new value generation.

### 24.6 IR-guided expert program

```cpp
transform ir<decomposition> add_lineage_split(
    ir<decomposition> alternatives)
{
    auto edit = ce::ir::rewrite(alternatives);

    edit.add(
        ce::decompose_by(lineage_components)
            .partials(lineage_sum)
            .fallback(ce::unsplit));

    return edit.commit();
}

field void lineage_response(...) <[
    offer transform add_lineage_split;
    offer cost_model machine_room_cost;
    inspect ir<semantic>, ir<decomposition>, ir<executable>;
::
    response = expression -[regulation]-> genes;
]>
```

The transform adds an exact alternative. It does not directly emit CUDA instructions.

## 25. Implementation-defined behavior

The following may be implementation-defined but MUST be documented and queryable:

- supported host C++ and CUDA revisions;
- default language revision for unversioned pragma;
- available semantic operation protocols;
- supported public IR schema versions;
- candidate and provider discovery mechanism;
- maximum automatically specialized profile alternatives;
- default planner objective and tolerance;
- empirical measurement policy;
- target capability vocabulary;
- native IR form;
- artifact formats and compatibility windows;
- graph-capture and asynchronous execution capabilities;
- supported compiler extensions and warning groups.

The following are language-defined and may not be changed by implementation policy:

- domain incompatibility;
- exact-coverage and contribution-ownership requirements;
- distinction among structure epoch, value generation, support generation, and order identity;
- output-update semantics;
- hard constraint and force behavior;
- no hidden canonicalization or synchronization;
- effect-contract conservatism;
- field optimization visibility;
- correctness independence from representative profiles.

## 26. Extension and versioning strategy

A source file may pin a language revision:

```cpp
#pragma cellerator 0.1
```

A future incompatible semantic change requires a new language revision.

Feature tests must permit conditional use:

```cpp
#if __has_cellerator_feature(ir_transform_v1)
    // ...
#endif
```

Experimental features should use versioned library namespaces or an explicit pragma extension:

```cpp
#pragma cellerator extension(ir_physical_write_v1)
```

A compiler MUST NOT silently reinterpret an unknown semantic directive as an ignored hint.

Public compiler-semantic protocols and IR schemas use adjacent versions. Persisted artifacts carry language, schema, target, toolchain, structure, order, and generation dependencies sufficient to reject stale reuse.

## 27. Rejected or avoided designs

### 27.1 Filename-selected semantics

Rejected because Cellerator must work in ordinary C++ compilation style, including headers and mixed projects. The pragma is the semantic switch.

### 27.2 A standalone non-C++ replacement language

Rejected for this proposal because it would discard C++ templates, pointers, libraries, CUDA interoperability, and the freedom to leave the abstraction.

### 27.3 CUDA-style launch geometry as the field meaning

Rejected. `<[ ... ]>` opens planning; it does not specify blocks, threads, devices, or one kernel.

### 27.4 Relation arrow as CSR, GEMM, or SpMM syntax

Rejected. The arrow expresses typed biological transfer. Physical candidates remain plural.

### 27.5 Forcing every operation under the relation arrow

Rejected. Support contraction, segment algebra, sparse updates, hierarchy operations, and effectful mutation have distinct semantics. They use compiler-visible library operations instead of decorative operators.

### 27.6 Bare arrow with hidden destination mutation

Avoided in revision 0.1. The relation transfer is an expression and C++ assignment states overwrite or accumulation.

### 27.7 Hard-coded biological ontology

Rejected. Human biological domains are useful nominal types and library declarations, but users may define arbitrary domains and relations.

### 27.8 Exposing current ABI structs as syntax

Rejected. Source concepts must correspond to stable semantics. Builders and versioned IR adapt to evolving records.

### 27.9 Hiding decomposition and IR permanently

Rejected. Expert access is a core product requirement.

### 27.10 Global cross-function planning without explicit fields

Rejected as a default. Fields communicate programmer-authorized optimization visibility. Ordinary compiler optimizations remain available outside them.

### 27.11 Treating unknown C++ calls as pure

Rejected. Unknown calls are opaque unless contracted.

### 27.12 Implicit canonicalization, synchronization, allocation, or assembly

Rejected. These operations have real performance and lifetime consequences and must be visible to planning and diagnostics.

### 27.13 Letting profile hints affect correctness

Rejected. Representative data selects among correct alternatives but does not establish exact semantics.

### 27.14 Unbounded branch multiversioning

Rejected as a default. The compiler joins states or uses bounded explicit alternatives.

### 27.15 `force` bypassing verification

Rejected. Invalid compiler plans are not a legitimate power feature. Fully manual C++ or CUDA is the unchecked escape hatch.

### 27.16 Hot-path dynamic plugin discovery

Rejected. Candidate catalogs are assembled before sealed execution.

### 27.17 A Fortran clone

Rejected. Cellerator may borrow explicit declarations and `::`-like visual separation, but C++ remains the host language and cultural base.

### 27.18 One universal atom

Rejected. Atomicity is relative to compiler level and algebra.

### 27.19 Biological identity for padding or storage frames

Rejected. Physical convenience never manufactures biological identity.

## 28. Open Design Questions

Only the following issues remain genuinely unresolved in this proposal:

1. **Intrinsic spelling versus recognized library types.** The semantic behavior of domains, axes, states, and relations is clear, but the final balance between compiler keywords and versioned `ce::` templates needs parser and tooling experiments.

2. **Expression-valued anonymous fields.** Revision 0.1 treats anonymous fields as compound statements. Allowing `<[ ... ]>` to produce a value could improve composition but may complicate C++ parsing and lifetime rules.

3. **Writable lower IR breadth.** Semantic and decomposition transforms should be public. The safe, stable writable surface for projection, packed, and executable IR needs experience across more targets.

4. **Profile dispatch policy.** The language model supports bounded alternatives, but default specialization limits, guard generation, and profile-join economics should remain implementation policy until measured.

5. **Exception semantics.** Preservation rules are specified, but a polished interaction among C++ exceptions, asynchronous work, and field-level planning needs a dedicated design pass.

6. **Privileged global IR.** The interface between core Cellerator IR and CellShard-owned evidence, basis, placement, persistence, and global scheduling should remain an adjacent extension rather than being frozen into language 0.1.

## 29. Research grounding

This proposal was derived from the Cellerator source tree rather than imposed on it.

The most influential source surfaces were:

- `include/Cellerator/compute/operation/operation_core_v2/schema.hh`
- `include/Cellerator/compute/operation/relation_algebra_v2/`
- `include/Cellerator/execution/biological_abi.hh`
- `include/Cellerator/execution/lifetimes.hh`
- `include/Cellerator/execution/launch_bindings.hh`
- `include/Cellerator/geometry/compiler/v2/`
- `include/Cellerator/execution/geometry_acquisition_v2/`
- `include/Cellerator/planner/end_to_end_planner.hh`
- `include/Cellerator/planner/portfolio/`
- `include/Cellerator/compute/decomposition/`
- `include/Cellerator/execution/joint_compiler/`
- `include/Cellerator/execution/object_binding/`
- `include/Cellerator/execution/atom_plane/`
- `include/Cellerator/execution/projection_value_plane/`
- `include/Cellerator/execution/lowering_resumption/`
- `include/Cellerator/compute/operation/candidate_catalog_v2.hh`
- `include/Cellerator/compute/operation/candidate_catalog_v3/`
- `include/Cellerator/planner/external_cost/`
- `include/Cellerator/execution/program/`
- `include/Cellerator/execution/training_program_v2/`
- `planning/jbc-preledger-v1/01_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md`

The source repeatedly establishes the same architectural facts:

- typed `operation_kind` problems, biological identity, and exact coverage are independent of physical shape;
- immutable structure, mutable values, support overlays, order, launch state, and evidence have separate lifetimes;
- complete plan cost includes packing, movement, order, synchronization, communication, and reuse;
- relation operations form a broader algebra than SpMM;
- exact decomposition, partial reconstruction, multi-extent binding, and persistent physical order are real compiler mechanisms;
- target-specific projection and executable recipes are lower than semantic meaning;
- manual providers and external costs are expected extension points;
- prepared execution must be sealed and allocation-free in the hot path;
- the compiler architecture already suggests a layered IR rather than one opaque lowering.

Relevant external precedents were used narrowly:

- C++ and modern Fortran informed declaration and block readability;
- Clang pragma and extension mechanisms informed source opt-in and effect syntax;
- CUDA provided a useful contrast between explicit launch geometry and an open planning envelope;
- MLIR informed typed multi-level IR, verification, and programmatic transformation;
- Clang/GCC inlining culture informed semantic inlining and natural access to compiler power.

No precedent determines Cellerator's semantics. The language is organized around Cellerator's own biological execution model.
