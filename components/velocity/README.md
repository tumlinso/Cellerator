# Cellerator Velocity

Use `import cellerator.velocity;` and link `Cellerator::velocity` for the C++20 RNA
velocity reference. Methods are available in `cellerator::velocity`. The import
includes the sensitivity and selective precision API in
`cellerator::velocity::precision`.

The CPU reference supports deterministic steady-state fitting, second-order
moments, stochastic refinement, and FP32 and FP64 storage and computation.
Its typed Cellerator inputs support signed dense strides, CSR, and CSC.
Callers supply topology, workspace, results, and output buffers. Static and
runtime replacement entry points let callers substitute stages or the complete
method.

Bind a native view with `cellerator::velocity::borrow_matrix` and use the result
in the method's `Input`. Keep the underlying buffers alive through preparation
and execution. For file-backed inputs, use `import cellerator.anndata;` to fill native buffers
before binding the method. See the [API](../../docs/development/velocity/api.md) and [precision](../../docs/development/velocity/precision.md)
guides for sizing, preparation, and numerical contracts.

Build and install from the Cellerator root using the
[single-cell methods guide](../../docs/development/single-cell-methods.md). The native
implementation comes from the tracked [CCell source](https://github.com/tumlinso/CCell/tree/codex/cellerator-native). See [validation](../../docs/development/velocity/validation.md) for the
source and toolchain associated with historical qualification reports.
