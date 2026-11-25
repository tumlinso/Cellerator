# Development

Build from the Cellerator root with `CELLERATOR_ENABLE_PREPROCESS=ON`.
Enable `CELLERATOR_BUILD_TESTS` for native checks and examples. See the
[single-cell methods guide](../single-cell-methods.md)
for build and server validation settings.

The numerical module accepts native Cellerator views. Consumers keep input
storage alive, provision output and workspace, and control parallelism.
Checked calls report status and required capacity. Trusted calls use the
documented preconditions. The reference allocates no memory during sizing,
validation, preparation, or execution.

The native qualification drivers use `CELLERATOR_BUILD_METHOD_QUALIFICATION`.
The Scanpy comparison driver also needs the optional AnnData component.
The historical [CCell source archive](https://github.com/tumlinso/CCell/tree/codex/cellerator-native)
records the original qualification scripts and their source context.
