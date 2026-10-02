# Architecture: one mathematical substrate, several kinds of meaning

This is a concrete target with adjustable names, not a requirement to replace the current architecture wholesale. The adoption task must reconcile it with the completed predecessor code. Source references [CE01–CE09, BP01–BP04, GH01–GH03] resolve in `evidence/source-ledger.json`.

## 1. Preserve and generalize the existing Cellerator split

The inspected source already separates semantic operations, candidate selection, physical projections, binding and execution. Its indexed-mechanism implementation already owns canonical parameters, asynchronous readers and saved forward tapes. Its training program selects prepared candidates rather than switching on model meaning at execution time. Build around these strengths.

Use three stages:

    describe logical computation → prepare a realization → bind and execute

**Description** names axes and domains, logical state/parameter identities, ordered arguments, operations, output contributions and requested capabilities. It can describe scalar state, actor-local hidden state, matrix patches, port interfaces, effect maps or supplied sparse relations. These are distinguishable data objects, not mandatory views of one flat biological ontology.

**Preparation** chooses state placement, operation cohorts, projection families, formats, numerical providers, derivative programs, workspaces and optional fusion. It can accept a caller-supplied strategy or already prepared realization. Preparation is allowed to allocate and validate; execution is not secretly required to repeat it.

**Execution** binds buffers and streams to prepared work. Values, parameter generations, activity and saved-primal lifetimes belong to established owners. Callers can explicitly choose outputs in a useful physical order instead of paying to return to canonical order after every operation. A native direct call remains available below the convenient composition interface.

Do not add separate semantic/packing/learning runtimes that all own the same coefficients. Choose one canonical parameter master and one ownership rule for physical copies. Reuse existing identity and error types where adequate; adapters in the supplied seed are witnesses, not replacement ABIs.

## 2. First-class mathematical objects, without freezing their shapes

The actor-by-private-state object records biological actor identity, each actor's local coordinate domain and extent, readout roles, structural support and the current incarnation of each slot. Uniform widths can expose a matrix view; ragged widths use offsets or width cohorts. The hidden columns of different actors need not have the same meaning. A plain scalar-state patch is also valid and must not acquire fictional hidden regulators because it is reshaped for a matrix operator.

Keep observable readout separate from hidden storage: a row may use a linear projection, a supported nonlinear readout or a declared explicit observable. A transpose is a physical/algebraic operation, not an automatic conversion of private coordinates into shared biological axes. Two-sided mixing is available when the selected model gives it meaning. Local ports give actors a communication interface without equating their private hidden bases.

Supports need tagged universes and provenance: measurement detection, declared primal dependencies, derivative dependencies, output contributions, source sequence intervals, and candidate nomination are different objects. Include universe identity and version with bitset/rank views. Zero numerical value does not mean absent dependency. Jaccard similarity proposes placement or a scientific investigation; it does not license state merging.

An operation advertises capabilities individually: forward, input VJP, parameter VJP, JVP, selected higher response, exact/differentiable solve, capture compatibility, deterministic accumulation and supported precision/shape/layout. Unsupported is a first-class result. A forward CUDA kernel does not grant an autograd implementation.

## 3. Packing is a strategy, a realization is not a strategy

The detailed contract is in `03_PACKING.md`. Preserve the generic two-sided occupancy oracle and the existing frozen Cellpack image as separate things. The inspected generic view admits independent row/feature orders; the frozen feature-first image is narrower. The existing CPK1 N=1 candidate is a legitimate specialized consumer, not a definition of all possible execution geometry.

A packer consumes semantic work and a cost/capability request and proposes one or more realizations. A lowerer validates and turns a chosen realization into native prepared candidates. A useful strategy may use no packing, independent source/destination permutations, op cohorts, hypergraph partitions, mixed dense/sparse regions or a caller's explicit plan. It need not emit Cellpack's block grammar. Provider-specific payloads are permitted with an explicit family identifier and validated semantic mapping, not a universal opaque blob that every consumer must understand.

## 4. Separate repository ownership from target-level dependencies

The current sequence bridge in Cellerator consumes Baseplane's exact predicate API. The newer Baseplane hierarchy consumes Cellerator's numerical providers. A naive package-wide dependency would create a cycle. Resolve it at the component level:

    Baseplane::seq                     Cellerator core/math/runtime
           \                            /                 |
            \---- optional sequence bridge --------------|
             \                                           |
              Baseplane representation/query components -|
                                                         |
                                    GlassHelix experiments

`Baseplane::seq` remains independently buildable. Cellerator core numerical components must not require Baseplane representation. The optional bridge links the exact-sequence core and Cellerator; it must not pull a package umbrella that recursively enables the reverse dependency. Baseplane's higher representation links exact sequence and selected installed Cellerator components. GlassHelix links installed Cellerator and optionally Baseplane only for experiments that use sequence.

Component names may follow the existing targets (`native_foundation`, `indexed_mechanism`, `local_differential`, etc.). Do not create a mandatory `Cellerator::everything`. Test each dependency direction with a clean out-of-tree installed consumer. Development superbuilds may compose projects explicitly; supported consumers must not compile sibling private sources. A release snapshot may pin three compatible commits without turning the three Todo databases into one authority.

## 5. Baseplane: source-grounded questions and representations

Unify exact sequence access, predicate evaluation, source spans, effect construction, hierarchy/refinement, candidate directories and learned representation into a usable composition surface. Not every pipeline must include every mechanism. A client should be able to use exact predicates alone, effect summaries alone, or a learned hierarchy with revisits.

A useful carrier names its source support, representation kind, value/weight/context versions and supported questions. Put large source interval sets, residual reservoirs and diagnostics off the hot descriptor; avoid a pointer-rich node object per base. Repeated content and repeated occurrence are separate identities. Shared exact content may share context-free computation; position, strand and cellular context remain per occurrence.

The representation may answer a query, return a certified bound for that specific query, or request detail. Learned confidence is not a certificate. Bounded wavefronts and explicit capacity/overflow policy preserve the ability to implement masks, warp collectives, compacted cohorts or ordinary branches. “Branch-free” is a useful machine technique, not a correctness requirement or a ban on multi-kernel execution.

Real nonlocal relationships require a real candidate directory. Warp equality checks only operate on already co-resident candidates. Keep full posting membership, handle large groups with factorized or tiled processing, and measure approximate nomination recall separately from final scoring. Invalidating a sequence edit may change both values and membership topology; these are separate generation events.

## 6. GlassHelix: an experiment with a strong library beneath it

Keep scientific specifications, data adapters/splits, measurements, hypotheses, fitted models, conditioning, perturbation protocols, evidence and interpretation in GlassHelix. Reusable derivative, dynamical, reduction, factorization, routing and state-transform mathematics lives in Cellerator. Inventory both native GH headers and Torch experiment code: either may contain general math worth extracting. A scientific likelihood's choice stays in GH; its reusable numerical evaluation may be a CE operation.

The existing `assembled_mechanism` already binds Cellerator operations and preserves ordered scientific roles. Do not replace this with a hidden scalar callback implementation when a registered native path exists. Consolidate current ML2 CellTag and moonshot models behind a small experiment runner and result format, not a new universal model framework.

Keep whole joint candidates intact through inference, resampling, interventions and interpretation. Shared computation is allowed when identical; recombining coordinates from incompatible hypotheses is not. State estimation remains necessary when current hidden values are unknown, even after a dependency or state variable has been made explicit.

## 7. Structural change is deliberate publication

An external scientific or engineering proposal supplies old epoch, new declarations, state/parameter/readout migration, initialization of new state, optimizer policy, evidence label and invalidated projections. Cellerator validates and executes the reusable transformation. GlassHelix decides what the proposal means scientifically. BP supplies sequence provenance where relevant.

Drain old readers/tapes or retain their valid old generations before publication. New slots get new incarnation IDs. A permutation permits corresponding optimizer-state permutation; a general dense basis change does not make diagonal Adam moments covariant. Use an explicit reset or an actually valid optimizer-specific transform. A rank reduction needs an approximation declaration or a verified invariant reduction, not just correlated activations. Preserve residual capacity and distinguish an initially function-preserving feed-forward branch from a new recurrent state that needs its own evolution law.

## 8. Physical organization follows ownership

Candidate destination map, reconciled at adoption:

- CE `include/Cellerator/{state,packing,math}` and matching source subdirectories for new cohesive APIs, only where existing `compute`, `geometry` or `execution` paths cannot already host them cleanly. Avoid a rename-only campaign. Existing providers stay under the established architecture/provider tree.
- BP `seq` remains the exact core; `representation`, `query`, `index`, `incremental` and `learning` hold sequence-facing composition, not duplicated general algebra.
- GH `experiment`, `data`, `models`, `analysis`, `intervention` and scientific `refactoring` hold consumer logic. Existing core/system types may remain if useful.

Each moved implementation has a source-to-destination map, repaired consumers and a single preferred entry path. Archive old experimental drivers after their useful behavior is reachable through the library; retain raw evidence and seed packages unchanged. A callable experimental candidate can live behind an optional component without being a second unmaintained tree.
