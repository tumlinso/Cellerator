# Biological dynamics examples

These small numerical systems demonstrate ways to compose the new packages.
Every biological label and rate below is an invented example; running the code
does not validate a biological hypothesis.
Examples using `torch.nn.Module` or Torch collectives require the optional
PyTorch dependency. The native conditional-dispatch example also requires its
optional compiled extension.

Run from the repository root with the source package:

```sh
PYTHONPATH=python python examples/framework/trajectory_experiment.py
PYTHONPATH=python python examples/framework/trajectory_training.py
PYTHONPATH=python python examples/framework/cellular_ai.py
PYTHONPATH=python python examples/framework/sparse_regulation.py
PYTHONPATH=python python examples/framework/spatial_population.py
PYTHONPATH=python python examples/framework/extracellular_coupling.py
PYTHONPATH=python python examples/framework/stochastic_history.py
PYTHONPATH=python python examples/framework/reaction_network.py
PYTHONPATH=python python examples/framework/shared_learned_dynamics.py
PYTHONPATH=python python examples/framework/coupled_population.py
PYTHONPATH=python python examples/framework/observation_inference.py
PYTHONPATH=python python examples/framework/scheduled_experts.py
# Requires the optional native conditional extension build:
PYTHONPATH=python python examples/framework/scheduled_native_experts.py
PYTHONPATH=python python examples/framework/response_precision.py
PYTHONPATH=python python examples/framework/distributed_experts.py
```

`trajectory_experiment.py` records observations and events as inputs change.
`trajectory_training.py` trains shared Torch dynamics over ragged private cell
states in explicit truncated-gradient windows.
`cellular_ai.py` composes sparse gene-addressed state, gated interactions,
cell-specific observations, and cell-to-cell signals in one illustrative CPU
model.
`sparse_regulation.py` makes a state change select gated relations from a
sparse interaction graph. `spatial_population.py` builds local neighbor
interactions and a graph-diffusion step. `extracellular_coupling.py` exchanges
amounts between cell identities and finite environment compartments.
`reaction_network.py` simulates identity-addressed discrete species counts
with sparse reaction hazards and a separate noisy observation.
`stochastic_history.py` runs
caller-defined stochastic channels and records event history plus observed
states. `shared_learned_dynamics.py` uses the optional Torch adapter to apply
one trainable rule to two independent cell states. `coupled_population.py`
routes a signal between cells and suspends/resumes the recipient.
`observation_inference.py` weights hidden-rate candidates under a supplied
observation score.

The scheduled-expert example connects explicit selected work to topology
estimates, exact parameter revisions, and a CPU callback, then commits results
to their original cells. The native conditional example uses the optional
compiled CPU dispatch adapter while preserving exact expert revisions and
original cell ownership. The response-precision example tests state overrides
against supplied finite response probes. The distributed-expert example shows
the optional Torch adapter in its single-process mode; a multi-process CPU
Gloo test is described in the
[distributed dispatch guide](../../docs/distributed_dispatch.qmd).

These scripts illustrate a composition route from caller-defined state changes
to active sparse relations and then to explicitly selected and physically
planned work. Cellerator does not infer event-to-expert meanings or biological
assumptions. Numerical choices, measurements, and interventions remain
explicit Python inputs so callers can replace them with their own formulation.

Additional subsystem references:

- [Modeling trajectory and state vocabulary](../../docs/biological_framework.qmd)
- [Revisioned conditional execution](../../docs/distributed_execution.qmd)
- [Native conditional planning and scheduling](../../docs/native_scheduling.qmd)
- [Revision-pinned CUDA parameter transfer](../../docs/framework_parameter_transfer.qmd)
- [Worklist regulatory gates](../../docs/framework_regulatory_gates.qmd)
- [Native CUDA runtime](../../docs/framework_cuda.qmd), compiled for SM70 but
  not yet executed on a GPU
- [Incremental sparse invalidation](../../docs/incremental_cellular.qmd)
- [Cell and environment coupling](../../docs/cellular_coupling.qmd)
- [Sparse stochastic reaction networks](../../docs/reaction_networks.qmd)
- [Native sparse relation attention](../../docs/framework_sparse_attention.qmd)
- [Native grouped linear operations](../../docs/framework_grouped_linear.qmd)
- [Packed-state embedding bags](../../docs/framework_packed_embedding.qmd)
- [Training independent trajectories](../../docs/trajectory_training.qmd)
- [Response-sensitive precision](../../docs/response_precision.qmd)
