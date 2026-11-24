# CCell Velocity

Use `import ccell.velocity;` and link `ccell::velocity` for the C++20 RNA
velocity reference. Methods are available in `ccell::velocity`. The import
includes the sensitivity and selective precision API in
`ccell::velocity::precision`.

The CPU reference supports deterministic steady-state fitting, second-order
moments, stochastic refinement, and FP32 and FP64 storage and computation.
Its typed Cellerator inputs support signed dense strides, CSR, and CSC.
Callers supply topology, workspace, results, and output buffers. Static and
runtime replacement entry points let callers substitute stages or the complete
method.

Bind a native view with `ccell::velocity::borrow_matrix` and use the result
in the method's `Input`. Keep the underlying buffers alive through preparation
and execution. For file-backed inputs, use `import cellerator.anndata;` to fill native buffers
before binding the method. See the [API](docs/api.md) and [precision](docs/precision.md)
guides for sizing, preparation, and numerical contracts.

Build and install from the Cellerator root using the
[CCell guide](../../README.md). Velocity is tracked CCell source, with its
provenance recorded there. See [validation](docs/validation.md) for the
source and toolchain associated with historical qualification reports.
