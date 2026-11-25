# Native single-cell methods

Enable `CELLERATOR_ENABLE_VELOCITY` for CPU RNA velocity and
`CELLERATOR_ENABLE_PREPROCESS` for CPU preprocessing. Both components accept
typed Cellerator views and caller-owned workspaces. Set
`CELLERATOR_ENABLE_ANNDATA` independently for backed reads into native buffers.
CUDA builds provide routed search through `Cellerator::neighbors` and
`<Cellerator/neighbors/exact_search.hh>`.

```cmake
find_package(Cellerator CONFIG REQUIRED COMPONENTS velocity preprocess)
target_link_libraries(my_app PRIVATE Cellerator::velocity Cellerator::preprocess)
set_target_properties(my_app PROPERTIES CXX_EXTENSIONS OFF)
```

```cpp
import cellerator.velocity;
import cellerator.preprocess;
```

See the [Velocity API](velocity/api.md),
[preprocessing contracts](preprocess/operations.md), and
[AnnData buffer guide](anndata.md).

The original CCell source license and notices are in
[docs/licenses/ccell](../licenses/ccell). Velocity ships its
[third-party notices](../licenses/velocity/THIRD_PARTY_NOTICES.md) with the
component. The former CCell [research notes](../research/ccell/compact-flow-methods.md)
and [historical validation](../history/ccell/refactor-validation.md) have their
own locations.

## Server validation

Configure separate build trees with Clang and matching `clang-scan-deps`.
Use Cellerator revision `17cb31cc` for the original implementation. Its exact
component tree is preserved on the
[CCell archive branch](https://github.com/tumlinso/CCell/tree/codex/cellerator-native).

| Configuration | Configure options |
|---|---|
| Original methods | `-DCELLERATOR_BUILD_CCELL=ON -DCCELL_BUILD_TESTS=ON -DCELLERATOR_ENABLE_CUDA=OFF -DCELLERATOR_ENABLE_ANNDATA=OFF` |
| Split methods | `-DCELLERATOR_ENABLE_VELOCITY=ON -DCELLERATOR_ENABLE_PREPROCESS=ON -DCELLERATOR_ENABLE_CUDA=OFF -DCELLERATOR_ENABLE_ANNDATA=OFF` |
| Core only | `-DCELLERATOR_ENABLE_VELOCITY=OFF -DCELLERATOR_ENABLE_PREPROCESS=OFF -DCELLERATOR_ENABLE_CUDA=OFF -DCELLERATOR_ENABLE_ANNDATA=OFF` |
| Routed search | `-DCELLERATOR_ENABLE_CUDA=ON` and the server GPU architecture |

Run the original `ccell-velocity-*` and `ccell-preprocess-*` tests against
the corresponding `cellerator-velocity-*` and `cellerator-preprocess-*` tests,
with `CELLERATOR_BUILD_TESTS=ON` in both builds. Check the FP16, blocks, and
Linux allocation suites explicitly. Compare the mixed-storage results,
selection, and statuses, the serial and worker-block execution paths, and the
stochastic selection barrier. Record existing failures on the original build.
Run `cellerator-neighbors-exact-search` on the CUDA build. To check backed
reads, enable AnnData++ with a storage backend and run
`cellerator-single-cell-anndata`.

Install the split to a fresh prefix. Configure `tests/single_cell/consumer`
with that prefix and the same compiler and scanner, then build and run its
CTest check. The consumer imports both method modules and checks that
discovering them does not load the optional AnnData target.
