# Developing Cellerator

Start with [`AGENTS.md`](AGENTS.md), [`scope.md`](scope.md), the
[architecture](docs/architecture.qmd), and the
[dated implementation record](docs/current_implementation.qmd). The
[developer reference](docs/developer_reference.qmd) describes current build
and validation routes. The [agent-facing routing guide](docs/development/routing.md)
and generated source map identify current owners, consumers, tests/evidence,
target destinations, and preservation dispositions.

Cellerator is an early high-performance PyTorch extension for cellular model
building. Prefer Torch for general tensors, standard layers, autograd, and
optimizers; add Cellerator operations where model structure changes the
mathematical contract or useful execution geometry. Keep biological
assumptions with scientific consumers. New operation families should have a
real contract and consumer, not merely architectural symmetry.

For every code change, trace the public native and Python entry points, build
targets, callers, tests, and source-bound evidence that the change affects.
Keep identity, support, routing, layout, values, parameters, and runtime
bindings distinct. Report correctness, scientific validity, package usability,
and measured performance separately. Do not infer GPU support or speed from
the presence of a CUDA source file.

Use the live Project Control workflow for task status and ownership. This
document and the README are not task trackers. Preserve frozen evidence bytes;
add updated interpretation in a dated wrapper or capability record.
