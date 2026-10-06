# Native product2 integration

The native operation evaluates `y[i] = k[i] * x[a[i]] * x[b[i]]` in FP32. The
installed `cellerator` package exposes product2 forward, VJP and JVP through its
`_native` extension; the optional `cellerator.torch` package provides the Torch
module and autograd adapter. Build and install instructions are in
[`docs/development/python-bindings.md`](../../../docs/development/python-bindings.md).

`test_product_adapter.py` exercises the installed Python/Torch surface against
independent formulas, finite differences, aliases, malformed inputs, optimizer
updates and first-order derivative limits. The C++ installed-consumer project
under `tests/learning_consumer` uses the optional `Cellerator::torch` component.

The records in `evidence/` and `capability.json` qualify the earlier standalone
adapter and its separate product library. Their hashes and logs remain unchanged
and do not qualify the absorbed package. Fresh installed-package and runtime
acceptance is pending the active Cellerator binding migration and root-owned
validation. The exact old test and evidence verifier are retained as dated source snapshots in `history/`; they are not current package gates.

No performance victory, biological fit, native optimizer ownership, CUDA Torch
tensor support beyond the active adapter contract, mixed precision, higher-order
differentiation or sanitizer qualification is claimed by this consumer migration.
