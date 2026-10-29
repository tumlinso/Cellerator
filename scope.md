# Cellerator scope

## Purpose

Cellerator is a high-performance PyTorch extension for building cellular
dynamical models. It adds reusable model operations and execution semantics
that account for biological structure: identities, relationships, repeated
organization, conditional activity, and reusable work. That structure can
improve interpretability and reveal computation that need not be repeated.
Whether a particular representation or implementation is faster remains an
operation- and workload-specific question.

Cellerator enables scientific work. It does not decide what a cell means, which
relationships are biologically valid, or which model a consumer should adopt.

## The boundary with PyTorch

PyTorch remains the general numerical and learning framework. Use Torch for
ordinary tensor arithmetic, conventional neural layers, autograd, optimizers,
and model composition when those express the computation well. Cellerator owns
specialized operations when cellular structure changes the mathematical
contract, useful execution geometry, data organization, or complete cost.

The `nn` namespace should grow from implemented, useful `torch.nn.Module`
compositions over Cellerator operations and ordinary Torch components. It is
not a promised catalogue of biological layers. Native C++ entry points also
remain first-class; Python integration should not replace unique native
contracts.

## What belongs here

Cellerator owns reusable, model-neutral machinery for:

- cellular entity, axis, order, relation, and structure identity;
- numerical operations and derivatives specialized by the structure of a
  modeled system;
- logical support, indexing, routing, and conditional work, with explicit
  capacity, overflow, and invalidation behavior where applicable;
- physical layouts and value/gradient mappings that preserve logical identity;
- preparation, reuse, and execution semantics with explicit lifetime and cost
  boundaries;
- native CPU/GPU implementations and thin PyTorch integration for those
  operations;
- experimental execution fields and compiler research, separated from the
  stable package surface while contracts are being established.

This is an open set of operation families. New operations should fit the same
ownership pattern rather than require a new top-level framework. Not every
operation needs sparsity, a custom kernel, or compilation.

## What remains elsewhere

- **PyTorch:** general tensor operations, standard neural layers, autograd,
  optimizers, and general model-training facilities.
- **Scientific consumers:** biological hypotheses, domain-specific model
  assumptions, their validation, and complete applications such as Karyoplast,
  Cytoplast, or Blastback.
- **AnnData and upstream single-cell libraries:** established data models and
  classical preprocessing, QC, normalization, and analysis workflows. AnnData
  is a required Python dependency today, but a direct Cellerator AnnData
  preparation adapter is not implemented. Scanpy and RAPIDS-singlecell remain
  upstream; this repository's classical routines have been testbeds.
- **Storage and delivery systems:** durable datasets, sharding, transport, and
  publication remain with their owning projects, such as CellShard.
- **Sequence-specialized representation:** intrinsically sequence-grounded
  functionality remains with Baseplane; Cellerator owns general numerical
  execution around reusable inputs.

## Maturity

Cellerator is early and not near complete. Native operations, sparse support,
indexing, routing, gating, Python/Torch adapters, tokenization, layout work,
and execution experiments exist at different levels of integration. Sparse
indexing, routing, and gating are promising early capabilities; they do not
establish a complete adaptive model runtime or a universal speedup.

Execution fields and compiler/IR work are experimental. The current execution
field composes a narrow prepared operation and does not provide JIT or live
recompilation. The implemented `nn` module set is small. Future operations
belong in reusable operation families first and in `nn` when parameter
ownership, composition, or a stable model-level interface warrants it.

Keep software correctness, scientific validity, package usability, and
measured performance as distinct claims. Each capability claim should identify
its implementation, public entry point, contract, relevant tests, and bounded
evidence. Where source paths or consumer ownership are not reconciled, document
the current route and mark the target as proposed.

## Scope test for new work

A proposed operation belongs in Cellerator when biological identity or shared
structure changes its useful mathematical operation or execution, and the
contract can be reused across models. Prefer ordinary Torch operations when
structure does not change the contract or execution opportunity. Keep model
assumptions in the scientific consumer, state distinct from parameters and
structure, and avoid creating another biology ontology, dataset workflow,
tensor system, or optimizer framework to house one operation.
