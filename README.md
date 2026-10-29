# Cellerator

**A computational framework for biological machine learning and cellular dynamics.**

Cellerator makes biological state, dynamics, observations, events, and
interactions computationally expressible. It aims to represent biological
structure directly and organize computations around a system's identities,
relationships, conditional activity, and geometry. The framework includes its
own native numerical machinery, specialized operations, and layouts; dense
tensor APIs are one option among several.

Cellerator can compose with PyTorch for learned modules, autograd, and familiar
training workflows. Torch is optional integration infrastructure, not the
definition or limit of the framework. Cellerator supports model-native dense,
sparse, ragged, relational, event-based, and hybrid state, including specialized
native execution. Consumer projects own their scientific hypotheses and
application-specific biological interpretation.

The new [biological modeling framework](docs/biological_framework.qmd)
describes the shared state/dynamics/context vocabulary and current package
entry points.

For an executable example of a persistent cell, changing inputs, observations,
and an emitted event stream, see
[`trajectory_experiment.py`](examples/framework/trajectory_experiment.py).
The [framework examples index](examples/framework/README.md) links additional
state, interaction, stochastic, Torch, population, and inference examples.

The library is early and not near complete. Native operations, Python/Torch
adapters, sparse indexing and support, tokenization, layout work, and execution
experiments have different interfaces and evidence. A source file is not by
itself a supported package capability. See the dated
[implementation record](docs/current_implementation.qmd) for current
boundaries and evidence.

## Start here

- [Getting started](docs/getting_started.qmd): install and exercise a current
  CPU operation.
- [Biological framework](docs/biological_framework.qmd): state, dynamics,
  observations, events, interactions, and composition.
- [Architecture](docs/architecture.qmd): product boundaries, core concepts,
  and current versus proposed source organization.
- [Scope](scope.md): what belongs in Cellerator and what remains with Torch,
  scientific consumers, or adjacent libraries.
- [Model building](docs/model_building.qmd): composing real operations and
  modules.
- [Optional ggml quantization](docs/ggml.qmd): the separately enabled CPU
  block-quantization bridge.
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

Karyoplast, Cytoplast, and Blastback motivate demanding reusable capabilities
for learned cellular dynamics, interacting populations, and reconstruction
from incomplete observations. They remain independent scientific consumers.
AnnData, Scanpy, and RAPIDS-singlecell remain important upstream tools for
established single-cell analysis.

The name joins “cell” and “accelerator.”
