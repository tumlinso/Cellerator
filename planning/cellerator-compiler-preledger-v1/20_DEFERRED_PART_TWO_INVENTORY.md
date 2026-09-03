# Explicit deferred Part Two inventory

Part Two is not a hidden tail of Part One.

## Deferred JIT work

- general runtime Cellerator source compilation;
- long-lived JIT compiler service;
- runtime profile specialization/recompilation;
- code cache eviction and live replacement;
- distributed JIT coordination;
- runtime compilation security/deployment model;
- dynamic linking of newly compiled device/host code beyond narrow same-compilation prelude transforms.

Part One may use bounded early host compilation of compiler transforms and may retain PTX for normal driver loading. Those do not constitute the full JIT product.

## Deferred CellShard application/runtime work

- full concrete application of Cellerator rules to arbitrary datasets;
- new atom-store format implementation beyond compiler migration necessities;
- deep ruleset-driven sharding/materialization;
- global placement optimization for live fleets;
- residency/lease/cache evolution;
- transport, RDMA, object-store, and distributed delivery;
- runtime schedule instantiation and recovery;
- deep Cellerator JIT integration;
- production distributed execution under the new ruleset seam.

## Narrow Part One reservations

Part One may define and test:

- immutable portable Cellerator ruleset export;
- generic materialization requirements;
- external complete-cost callbacks;
- Cellerator external bindings and lowering resumption;
- compatibility adapters for existing embedded CellShard tests;
- a no-code-loss migration split between compiler semantics and concrete runtime/storage.

These reservations must not block final Part One completion on a new CellShard runtime.

## Explicitly not deferred

The following old JBC areas are compiler work and therefore Part One:

- evidence/proposal discovery;
- exact certification;
- atom semantics;
- composition grammar;
- basis/no-basis;
- global operation/program IR;
- portable schedule/ruleset compilation;
- decomposition and partial-result algebra;
- candidate and cost planning.
