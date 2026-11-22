# Working on CCell preprocessing

Read the [CCell contributor guide](../../AGENTS.md) and the root protected
C++ coding style. This directory owns the compiled CPU reference and the
`ccell.preprocess` module. Keep module exports small and link consumers to
`ccell::preprocess`.

Use typed Cellerator views with explicit shape, signed element strides where
needed, and separate sparse index and offset types. Preserve each operation's
input interpretation, precision, output order, and validation contract.
Callers retain input and output storage and provide scratch. Keep format
acquisition outside the methods. Resolve representation dispatch outside
numerical loops.

Build and validate through the [CCell guide](../../README.md). The
[numerical contracts](../../docs/preprocess/operations.md) describe the
reference behavior. Use the pinned Scanpy cases to assess consequential
numerical differences with scientific quantities and tolerances. Preserve
historical receipts and source-bound qualification records.
