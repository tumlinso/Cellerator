# Working on CCell Velocity

Read the [CCell contributor guide](../../AGENTS.md), root protected C++ coding
style, and the local [Velocity style record](STYLE.md). This directory owns
steady-state and stochastic RNA velocity methods. Its public CCell import is
`ccell.velocity` and its installed target is `ccell::velocity`. The underlying
`velocitypp` and `velocitypp.precision` modules supply the method types.

Bind typed Cellerator views and retain their underlying storage through
preparation and numerical use. Preserve signed dense strides, CSR/CSC
orientation, independent sparse index and offset types, selected feature
order, and explicit FP32 and FP64 contracts. Callers own topology, workspace,
results, and synchronization. Keep AnnData++ acquisition and conversion at the
optional adapter boundary.

Read the [API](docs/api.md), [architecture](docs/architecture.md), and
[precision](docs/precision.md) before changing public behavior. Validate from the
Cellerator root through the [CCell build guide](../../README.md). Keep historical
JSON receipts and qualification artifacts bound to their original source and
toolchain.
