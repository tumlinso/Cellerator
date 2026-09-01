# Cellerator integration-path and registry ownership

## Authority cursor

This is the `CE-JBC-B06` integration reservation at Cellerator Git commit
`81e2e17ca1b9aa8f5ba8798a21115fa1dac2c1e0`, registered CellShard source
commit `96a691e4a271fabd738ff5819eef6349ac3621a0`, and separately observed
Cellerator Todo revision `3614`. Git and Todo cursors are independent; Project
Control observations are not globally atomic.

The exact machine-readable reservation is
[`planning/jbc/cellerator_integration_only_paths_v1.csv`](../../planning/jbc/cellerator_integration_only_paths_v1.csv).
This bootstrap record is the only change: it does not edit a reserved path,
register a provider, advance the parent gitlink, or change build/runtime
behavior.

## Exclusive owner

After bootstrap, the Cellerator integration-only owner is
`CE-JBC-L-VERIFY-INTEGRATE`, acting through an explicitly claimed integration
task with path scope. The final planned aggregation task is `CE-JBC-V06`.
Documentation of ownership does not expand that task's live capability: if a
reserved path is absent from its Project Control scope, the integrator must
obtain an authority-approved scope update rather than editing it opportunistically.

Ordinary interface, decomposition, fragment, plane, resumption, cross-operation,
external-cost, provider, test, benchmark, and CellShard lanes must not write the
listed paths. They publish isolated source-linked fragments in their own task
scopes. Integration applies those fragments only after interface, source,
validation, and dependency receipts agree.

## Reserved path classes

### Root and subsystem target aggregation

`CMakeLists.txt` and the canonical subsystem CMake files assemble the native
target graph. Provider tasks may add a scoped target fragment or source
inventory beside their implementation, but only the integration lane includes
that fragment in root or subsystem aggregation. This prevents independent
lanes from racing target names, options, dependencies, package visibility, CUDA
policy, or standalone/embedded defaults.

### Provider and catalog aggregation

The provider-policy/target manifests, public provider contract and registry,
builtin catalog, CE-GEO assembly, CE-EXOP portfolio, relation-algebra catalog,
and candidate discovery/inventory are central authorities. A provider lane
exports a stable source-linked registration function and descriptor array; it
does not append itself directly to a registry or catalog. Experimental
candidates remain marked measurement-required and never self-promote.

### Public umbrella headers

Umbrella headers aggregate already validated versioned contracts. A provider
or interface lane creates its focused public header and standalone include
fixture without adding it to an umbrella. Integration adds an include only
after the version/hash, ownership, dependency direction, static layout, and
validator receipts pass. Frozen v1/v2/v3 records are not changed to avoid an
umbrella update.

### Package exports

The live tree has no `cmake/CelleratorConfig.cmake.in` or
`cmake/CelleratorConfigVersion.cmake.in`. Those exact future source locations
are reserved now. Any `install(EXPORT ...)`, package target, config template,
or compatibility-version addition is integration-only and must prove a clean
standalone downstream consumer. A provider lane cannot create a private package
surface that bypasses the canonical target graph.

### Component documentation and parent gitlink

The shared component charter and adapter documentation are integration-owned
because they describe dependency direction across repositories/frameworks.
Provider-specific notes remain with provider code.

The `components/CellShard` gitlink is also integration-only. CellShard is
committed and pushed first; the Cellerator integrator advances the pointer once
per checkpoint bundle, records both authority cursors, and reruns standalone
and embedded builds. Leaf/provider tasks never advance the parent pointer. The
currently newer registered checkout is therefore intentionally not absorbed by
this bootstrap task.

## Required source-linked fragment receipt

Every fragment presented to integration records:

1. JBC task, run, lane, role, source repository, commit, and clean-worktree
   observation;
2. exact added/changed source paths and content hashes;
3. consumed and produced interface versions, hashes, owners, and dependency
   direction;
4. stable operation, provider, candidate, projection, mechanism, and artifact
   identities where applicable;
5. caller-owned capacities or declared cold allocations, asymptotic bound, peak
   memory, candidate count, and explicit overflow diagnostics;
6. exact independent validator, reference/differential test, malformed-input
   test, property test, build, sanitizer, and benchmark commands applicable to
   the mechanism;
7. complete cost phases, reuse assumptions, hardware/build identities, and
   promotion/non-promotion disposition for measured work;
8. proof that no reserved central path, unrelated task, frozen wire semantic,
   or foreign repository policy was mutated;
9. downstream consumer or mirrored cross-authority receipt when required; and
10. the requested central aggregation action and safe rollback/compatibility
    route.

A receipt may explicitly state that a category is not applicable, with a
reason. It may not omit identity, exact coverage, allocation/capacity, or
validation information that the fragment actually uses.

## Integration checks

The integration lane independently verifies fragment hashes and interface
compatibility, rebuilds central catalogs from source-linked fragments, rejects
duplicate stable identities, and runs focused plus standalone/embedded build
matrices. It confirms that native libCellerator remains CellShard-independent,
that framework adapters remain adapters, and that enabling CellShard adds a
higher-level compiler/runtime without changing canonical `Cellerator::runtime`.

Exact biological coverage and canonical recovery are checked independently
before execution. Approximate evidence may propose a fragment but cannot certify
it. Proposal overlap, physical overlap, and execution contribution remain
separate; each logical contribution has one exact owner unless a versioned
partial-result algebra proves reconstruction.

## Classification

All live paths in the reservation are **preserve** as integration authorities.
The two absent package templates are **adjacent extension** reservations. The
parent CellShard pointer follows the **migrate** rule only at checkpoint-bundle
integration; no ordinary task may update it. Nothing in this reservation is
authorized for retirement.
