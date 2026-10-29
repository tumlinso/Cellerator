# Cellerator scope

## Purpose

Cellerator is a general machine-learning and computational framework for
biological modeling, with a particular emphasis on cellular dynamics. It
provides modeling abstractions, native numerical machinery, specialized
operations and execution paths for systems whose state and dependencies have
biological structure. Computation may use dense tensors, sparse relations,
ragged collections, event histories, spatial neighborhoods, structured objects,
or combinations of these.

Biological geometry is a computational opportunity. Cellerator can represent
identity, relations, dynamic support, state-dependent activity, lineage, and
spatial organization directly, and may select layouts or operations that exploit
them. No single layout, device, programming interface, or model family defines
the scope. Efficiency and biological validity require evidence for the
particular model and workload.

Cellerator enables scientific work. It does not decide what a cell means, which
relationships are biologically valid, or which model a consumer should adopt.

## PyTorch and other foundations

PyTorch provides a valuable optional ecosystem for ordinary tensor arithmetic,
conventional neural layers, autograd, optimizers, and model composition. Use it
where useful. Cellerator also owns general mathematical/modeling abstractions,
structured representations, numerical operations, and native CPU/GPU execution.
Torch adapters do not constrain the implementations those abstractions can use.

`torch.nn.Module` compositions are one route for learned submodels and
autograd. Native C++ entry points and specialized kernels are first-class.
Other state and computation backends can participate through explicit model
interfaces without being coerced through dense Torch tensors.

## What belongs here

Cellerator owns reusable, model-neutral machinery for:

- opaque cell instances with model-supplied persistent state and behavior;
- state, updates, observations, context, trajectories, and explicit state or
  event ownership contracts;
- cellular entity, axis, order, relation, lineage, and structure identity;
- numerical operations and derivatives specialized by the structure of a
  modeled system;
- logical support, indexing, routing, and conditional work, with explicit
  capacity, overflow, and invalidation behavior where applicable;
- physical layouts and value/gradient mappings that preserve logical identity;
- preparation, reuse, and execution semantics with explicit lifetime and cost
  boundaries;
- native CPU/GPU implementations, specialized kernels, and integrations with
  Torch or other numerical systems;
- model composition and useful inference methods that can be expressed against
  shared dynamical and observation contracts.

This is an open set of operation families. New operations should fit the same
ownership pattern rather than require a new top-level framework. Not every
operation needs sparsity, a custom kernel, or compilation.

## What remains elsewhere

- **PyTorch:** reusable tensor, neural network, autograd, and optimizer
  facilities that Cellerator models may compose with.
- **Scientific consumers:** biological hypotheses, domain-specific model
  assumptions, their validation, and complete applications such as Karyoplast,
  Cytoplast, or Blastback.
- **AnnData and upstream single-cell libraries:** established data models and
  classical preprocessing, QC, normalization, and analysis workflows. The
  AnnData adapter requires explicit input selection and does not preprocess or
  choose scientific semantics automatically. Scanpy and RAPIDS-singlecell stay
  upstream for established analysis workflows.
- **Storage and delivery systems:** durable datasets, sharding, transport, and
  publication remain with their owning projects, such as CellShard.
- **Sequence-specialized representation:** intrinsically sequence-grounded
  functionality remains with Baseplane; Cellerator owns general numerical
  execution around reusable inputs.

## Maturity

Cellerator is early and not near complete. Native mathematical operations,
model and state APIs, sparse support, indexing, routing, gating, tokenization,
layout work, and execution experiments have different maturity and validation.
Sparse indexing, routing, and gating do not by themselves establish a complete
adaptive runtime or universal speedup. Newly added modeling packages are rapid
development interfaces and require broader validation and evolution.

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

A proposed capability belongs in Cellerator when it advances reusable
biological modeling or takes advantage of meaningful structure in state,
dynamics, relations, observations, or computation. A capability can be useful
even when it is not sparse or faster than a standard library. Keep application
hypotheses and experimental interpretation with the scientific consumer;
provide flexible computational means to express and test them. Keep model
meaning, state, parameters, relations, and physical execution distinct enough
that different consumers can combine them deliberately.
