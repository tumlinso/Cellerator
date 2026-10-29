# Cellerator

**A high-performance PyTorch extension for building cellular models.**

Cellerator adds model operations and execution tools shaped around biological
structure. Repeated relationships, stable identities, sparse support and
conditional activity can guide both the mathematics and the way work is
organized for CPU and GPU hardware. The goal is to make cellular dynamical
models easier to inspect and more efficient by representing them closer to the
systems they describe.

Cellerator builds on PyTorch. PyTorch supplies tensors, ordinary numerical
operations, `torch.nn.Module`, autograd, optimizers and model composition.
Cellerator supplies specialized operations and execution semantics where the
structure of a cellular model changes the useful computation. It is an
extension to Torch, not a replacement for it, and it enables scientific work
without choosing a consumer's scientific assumptions.

The library is early and not near complete. Native operations, Python/Torch
adapters, sparse indexing and support, tokenization, layout work, and execution
experiments have different interfaces and evidence. A source file is not by
itself a supported package capability. See the dated
[implementation record](docs/current_implementation.qmd) for current
boundaries and evidence.

## Start here

- [Getting started](docs/getting_started.qmd): install and exercise the current
  CPU Python/Torch operation.
- [Architecture](docs/architecture.qmd): product boundaries, core concepts,
  and current versus proposed source organization.
- [Scope](scope.md): what belongs in Cellerator and what remains with Torch,
  scientific consumers, or adjacent libraries.
- [Model building](docs/model_building.qmd): composing real operations and
  modules.
- [Operations](docs/operations.qmd): current operation families and their
  maturity.
- [Layout and execution](docs/layout_execution.qmd): identity, support,
  routing, layout, preparation, and runtime boundaries.
- [Experiments and history](docs/experiments_and_history.qmd): compiler,
  execution-field, tokenization, and source-bound records.
- [Python bindings](docs/development/python-bindings.md): import surfaces and
  optional native build requirements.
- [Developer reference](docs/developer_reference.qmd): source/build entry
  points and validation commands.
- [Implementation style](docs/development/implementation_style.md): maintained
  contributor conventions.

Projects such as Karyoplast, Cytoplast, and Blastback own their models and
scientific interpretation. AnnData remains a required Python dependency and a
standard scientific data interface, but Cellerator does not yet provide a
direct AnnData preparation adapter. Scanpy and RAPIDS-singlecell remain the
upstream home for established single-cell analysis workflows; their routines
in this repository have served as testbeds and consumers.

The name joins “cell” and “accelerator,” with a “cell-torch” pun:
cell-(ara)-tor(ch).
