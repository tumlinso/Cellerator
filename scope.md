# Cellerator scope

## Purpose

Cellerator is an extensible machine-learning framework designed around the
computational structure of the cell and the systems it comprises. Its governing
question is:

> What would machine learning look like if its computational abstractions were
> developed around the structure of the cell and the systems it comprises,
> rather than cellular systems being translated into abstractions originally
> developed for language?

It aims to make learning cellular representations, relationships, and dynamics
as natural as learning sequences is in language-model frameworks, without
compromising general neural-network expressiveness and while exploiting
biological structure for computational efficiency. This biological ML ambition
is the authority for interpreting the framework's scope and later ecosystem
analogies.

PyTorch makes tensors and differentiable operations accessible. Sequence-model
libraries make sequence architectures convenient to construct. Cellerator aims
to make cellular systems equally natural computational subjects by offering a
richer vocabulary for representations, learned interactions, architectures,
and learning objectives. It does not prescribe what mathematical object a cell
must be.

Biology should inform representation, architecture, learning objectives, and
execution together. Molecular identities, conditional interactions, overlapping
regulatory modules, cellular history, and shared computations can inspire
reusable neural abstractions and efficient physical implementations. Compilers,
kernels, scheduling, and model distribution support this ambition; none alone
defines what makes Cellerator distinctive.

Dynamics remains a central motivation. Making explicit state transitions
optional preserves alternative ways to learn cellular behavior, including
history-dependent computations with distributed or opaque representations.
It does not privilege static encoders or require every cell model to simulate
biological timesteps.

The motivation is to develop biological learning problems on their own terms.
A ranked transcriptional profile need not contain everything required to
predict cellular behavior, and a masked-gene objective does not by itself
constrain the dynamics responsible for expression. Such representations and
objectives remain useful choices for appropriate questions. Attention remains
a general mechanism for learned, context-dependent interactions; neither
transformers nor sequence representations are excluded. The framework should
make alternatives and combinations equally natural to explore.

## Expressiveness, learning, and efficiency

- **Expressiveness:** make sophisticated biological representations and
  relationships easier to describe while preserving general neural-network
  capabilities. Dense networks, ordinary attention, transformers, recurrent
  models, sparse conditional computation, mixtures of experts, dynamical
  systems, and their combinations all belong in scope. This is practical
  modeling expressiveness, not a claim of a larger universal-approximation class.
- **Learning:** make biological questions accessible through architecture and
  objective design. Representations, regulatory interactions, histories,
  trajectories, perturbation responses, and conditional behavior can be
  learning targets or context, without requiring every model to represent them
  explicitly or use the same training objective.
- **Efficiency:** exploit structure that a model contains or discovers,
  including persistent information, overlapping modules, sparse dependencies,
  geometry, and repeated computational patterns. Structure should make computation
  more efficient or improve the model, rather than impose an expensive biological metaphor.

These ambitions should reinforce one another. Sparsity is an opportunity, not
a requirement: dense attention may be the right operation. A model need not
use `Cell`, explicit state transitions, biological annotations, or a particular
layout to participate. Computation may use dense tensors, sparse relations,
ragged collections, event histories, spatial neighborhoods, structured objects,
or hybrids. Biological structure may be supplied, learned, or left implicit;
known regulatory relationships are not mandatory constraints.

## Scientific questions and identifiability

Explicit identities and context-dependent interactions can make a biological
question clearer in the architecture and in what its objective requires the
model to learn. That may lower the operational barrier to investigating
identifiable components. It does not make a learned interaction causal or a
latent variable molecularly identifiable merely because it has a biological name.

Identifiability is relative to a specified question, admissible model family,
assumptions, and observation experiment. Appropriately sampled dynamics,
natural excitation, lineage, interventions, or other forcing may constrain an
answer within a defensible model class. A dynamical feature can be identifiable
without uniquely identifying every latent coordinate or biochemical
realization; latent-state ambiguity is not a universal impossibility result.

Cellerator supplies architectures and learning machinery through which these
questions and constraints can be expressed. GlassHelix investigates what
mechanism or feature can formally be inferred under specified conditions.
Scientific consumers own those conditions, their interpretation, and validation.

## PyTorch and other foundations

PyTorch provides a valuable optional ecosystem for ordinary tensor arithmetic,
conventional neural layers, autograd, optimizers, and model composition. Use it
where useful. Cellerator also owns general mathematical/modeling abstractions,
structured representations, numerical operations, and native CPU/GPU execution.
Torch adapters do not constrain the implementations those abstractions can use.

`torch.nn.Module` compositions are one route for learned submodels and
autograd. Native C++ entry points and specialized kernels are first-class.
Other representation and computation backends can participate through explicit
operation and model interfaces without being coerced through dense Torch
tensors or a universal state-transition interface.

## C++ and Python as modeling languages

C++ and Python are first-class languages for building, training, extending, and
executing biological models. The intended C++ modeling interface should be at
least as expressive as the Python interface for Cellerator's biological and
neural abstractions. It should make a complete model natural to author in C++,
including custom learned components, relationships, objectives, and readouts.
Native entry points for kernels alone do not meet this ambition.

The C++ experience should feel native to C++: its modeling API is designed
directly around idiomatic C++ types, composition, ownership, and extension.
It is not a translation of the Python API or a binding-shaped frontend.
Reusing LibTorch machinery does not mean adopting its frontend or reproducing
Python module conventions in C++. Shared native mechanisms are C++ library
facilities in their own right; Python can adapt them while retaining its own
useful modeling and extension capabilities.

C++ should foreground typed composition, inheritance where useful, generic
programming, explicit ownership, specialized representations, and integration
with HPC applications. Python should foreground interactive exploration,
dynamic composition, data workflows, and reuse of its scientific and ML
ecosystem. Both should support serious modeling and training. Comparable
expressiveness does not require identical syntax or duplicating every
language-specific ecosystem feature.

The C++ interface targets C++ model developers and HPC applications. Its native
style includes ordinary result structs, templates, primitive borrowed views,
explicit extents and lifetimes, and separate owners for parameters and buffers.
Named module registration, member enumeration, and a universal learned-model
base are not required. Names still carry biological meaning at data and saved
declaration boundaries. Optional host C++20 modules organize imports while CUDA
and conventional header builds remain independent.

Reuse established numerical and differentiation infrastructure, including
LibTorch where appropriate, while designing an intuitive Cellerator modeling
experience. Users should not need to understand backend registration and
resource plumbing to express a biological model. Native use should not require
a Python interpreter; Python models should retain direct access to ordinary
Python modules and custom operations. These are architectural requirements,
not a claim that the current interfaces have reached parity. See the
[language design direction](https://github.com/tumlinso/Cellerator/blob/main/docs/architecture.qmd#cpp-and-python-modeling).

## What belongs here

Cellerator owns reusable, model-neutral machinery for:

- learnable biological representations, interaction mechanisms, and neural
  compositions that remain compatible with ordinary neural-network operations;
- reusable learning objectives and training mechanisms for biological
  questions, with explicit observation, conditioning, and supervision contracts;
- reusable model definitions, configuration and serialization contracts,
  pretrained loading, and sharing interfaces that preserve biological feature
  identities, input meaning, and dependencies of attached readouts;
- learned or supplied relationships, contextual attention, overlapping modules,
  and dense or conditional computation;
- cellular entity, axis, order, relation, lineage, and structure identity;
- optional persistent state, cell instances, updates, observations, histories,
  trajectories, and event ownership for models that need them;
- numerical operations and derivatives specialized by the structure of a
  modeled system;
- logical support, indexing, routing, and conditional work, with explicit
  capacity, overflow, and invalidation behavior where applicable;
- physical layouts and value/gradient mappings that preserve logical identity;
- preparation, reuse, and execution semantics with explicit lifetime and cost
  boundaries;
- native CPU/GPU implementations, specialized kernels, and integrations with
  Torch or other numerical systems;
- model composition and useful inference methods with explicit representation,
  observation, and learning contracts; dynamical contracts apply where relevant.

This is an open set of operation families. New operations should fit the same
ownership pattern rather than require a new top-level framework. Not every
operation needs sparsity, a custom kernel, or compilation.

## A coherent ecosystem for cell models

Building, training, and sharing cell models is the user experience that serves
the biological ML ambition above. The Transformers analogy describes the
coherence and usability of that experience, not a replacement for cellular
computational abstractions or a requirement for transformer architectures,
sequence tokenization, PyTorch, or one mathematical definition of a cell.

Reusable model components, biological input representations, learning objectives,
trained parameters, and supported readouts should fit together through minimal,
consistent interfaces. A researcher should be able to load a cell model,
understand the questions it addresses, and extend it with another learned
readout without reconstructing the original research codebase. Static encoders,
event-conditioned models, recurrent or dynamical models, and population models
all belong in this ecosystem.

Hugging Face Hub integration and possible library registration are subordinate,
prospective interoperability milestones. They would support distributing and
discovering models; they do not define Cellerator's architectures, establish scientific
validity, or demonstrate performance. This is product direction, not a claim
of an existing integration or authorization to implement or register one.

## What remains elsewhere

- **PyTorch:** reusable tensor, neural network, autograd, and optimizer
  facilities that Cellerator models may compose with.
- **Scientific consumers:** biological hypotheses, domain-specific model
  assumptions, their validation, and complete applications such as Karyoplast,
  Cytoplast, or Blastback.
- **GlassHelix:** formal investigation of what mechanisms or features are
  inferable for a specified question, model family, assumptions, and experiment.
  Reusable learning and numerical machinery can still belong in Cellerator.
- **AnnData and upstream single-cell libraries:** established data models and
  classical preprocessing, QC, normalization, and analysis workflows. The
  AnnData adapter requires explicit input selection and does not preprocess or
  choose scientific semantics automatically. Scanpy and RAPIDS-singlecell stay
  upstream for established analysis workflows.
- **Storage and delivery systems:** durable dataset storage, sharding, transport,
  and hosting services remain with their owning projects, such as CellShard or
  model hubs. Cellerator owns the model save/load and sharing interfaces that
  connect to those services; it need not build another hosting platform.
- **Sequence-specialized representation:** intrinsically sequence-grounded
  functionality remains with Baseplane; Cellerator owns general numerical
  execution around reusable inputs.

## Maturity

Cellerator is early and not near complete. Learned components, training
interfaces, native mathematical operations, model and state APIs, sparse
support, indexing, routing, gating, tokenization, layout work, and execution
experiments have different maturity and validation.
Sparse indexing, routing, and gating do not by themselves establish a complete
adaptive runtime or universal speedup. Newly added modeling packages are rapid
development interfaces and require broader validation and evolution.

Execution fields and compiler/IR work are experimental. The current execution
field composes a narrow prepared operation and does not provide JIT or live
recompilation. The implemented `nn` module set is small; this is an implementation
boundary, not a ceiling on the framework's learning capabilities. Grow reusable
architectures and objectives alongside the numerical mechanisms they need,
reusing existing neural-network infrastructure where it serves the model.

Keep software correctness, scientific validity, package usability, and
measured performance as distinct claims. Each capability claim should identify
its implementation, public entry point, contract, relevant tests, and bounded
evidence. Where source paths or consumer ownership are not reconciled, document
the current route and mark the target as proposed.

## Scope test for new work

A proposed capability belongs in Cellerator when it improves the construction,
training, execution, reuse, or sharing of models that learn biological systems.
Begin with the governing biological ML question: identify which biological
representation, relationship, behavior, or learning problem becomes easier to
express, which constraints the objective supplies, and which structure execution
can exploit. Evaluate model portability and sharing as support for that purpose.
A capability can be useful without being sparse, dynamical, or faster than a
standard library. Do not make a static representation learner or a dense
network adopt a cell lifecycle merely to use the framework.

Keep application hypotheses and experimental interpretation with the scientific
consumer. Preserve the distinctions among observations, learned representations,
optional persistent state, parameters, relationships, and physical execution so
different model families can combine them deliberately. The architecture and
objectives should reflect the biological question without prescribing its answer.
