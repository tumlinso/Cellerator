# CelleraTorch experimental workspace

This directory is a place to explore a simpler, faster, maintainable biological
numerical runtime around Cellerator. It is an experiment in the repository's
shared Git history, not a CMake component, package, or independent worktree.
Nothing here currently implements or qualifies that runtime.

Implementation is on hold until the user explicitly starts the experiment.
The current scope is this folder and its documentation only.

The intended division of work is:

- PyTorch provides orchestration, scheduling, autograd, and graph execution.
- Cellerator provides Torch-independent native kernels, flexible internal
  representations, and optimizations that use biological structure.

The aim is to reuse PyTorch and Cellerator infrastructure wherever it already
fits, and to simplify around demonstrated needs rather than recreate framework
services. See [the design note](docs/design.md) for boundaries and evidence
guidance.

This workspace is distinct from the existing optional `python/cellerator/torch`
adapter and `bindings/torch` integration, and from the historical
`components/CelleraTorch` component described in dated records. Those paths
describe existing or historical integration; this folder records an
experimental direction only.
