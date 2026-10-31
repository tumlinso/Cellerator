# Python bindings

The distribution is named `cellerator`. Python and Torch are adapters over
specific Cellerator contracts; neither adapter is a general replacement for
the native API. The package is early and exposes operations with different
backend and maturity boundaries. See the [source routing map](routing.md) and
[current implementation record](../current_implementation.qmd) before relying
on a capability.

AnnData is a required dependency (`anndata>=0.10`). The
`cellerator.adapters.anndata` adapter selects explicit `X`, layer, or raw inputs,
preserves sparse data, and aligns cells and output observations by identity.
Numerical entry points retain their explicit array/tensor contracts. Scanpy and
RAPIDS-singlecell remain upstream for classical analysis and preprocessing.

## Modeling without a native extension

`Cell`, `DynamicalCell`, `DynamicalModel`, `StepContext`, `StepResult`, and `simulate` are available from
`cellerator` without loading the compiled numerical extension or Torch. The
`state`, `events`, `interactions`, `dynamics`, `execution`, `inference`, and
`spatial` packages provide model-independent reference mechanisms. State can be
an arbitrary object or an external state reference; observations and update
rules are supplied separately for dynamical models. Generic `Cell` handles bind
arbitrary information to shared callable programs without a clock or mandatory
state vector. The legacy `Model` name aliases `DynamicalModel`.
See the [neural foundation](../nn_foundation.qmd) for composable learned
features, relations, readouts, and ordinary parameter-owned custom operations,
and the [framework guide](../biological_framework.qmd)
for examples and the boundary between these mechanisms and native execution.

Optional adapters are explicit imports: `cellerator.adapters.torch`,
`cellerator.adapters.distributed`, `cellerator.adapters.cuda`, and
`cellerator.adapters.ggml`. A native host build additionally provides
`_modeling_native` and `_conditional_native`; the new CUDA runtime and ggml
bridge are opt-in CMake components. Importing a modeling package does not imply
that those compiled components are installed.

## Host Python and Product2

The default Python build enables the `_native` extension and native Product2,
with CUDA and the compiled Torch adapters disabled. Install from the repository
root with:

```sh
python -m pip install .
```

The `cellerator.ops` module exposes native CPU Product2 preparation, forward,
JVP, and VJP using NumPy FP32 arrays. `cellerator.torch.product2` exposes
`product2`, `product2_jvp`, and `Product2Module`. The module is a real
`torch.nn.Module`: it owns coefficients as a Torch parameter, retains packet
indices as buffers, prepares topology for the input extent, and supports
first-order input/coefficient gradients. Inputs and outputs are CPU FP32
vectors; packet indices are int64 in the Torch adapter. Higher-order gradients
and a GPU Product2 path are not claimed.

```python
import torch
from cellerator.nn import Product2Module

op = Product2Module(
    torch.tensor([0.5, 1.25], dtype=torch.float32),
    torch.tensor([0, 1], dtype=torch.int64),
    torch.tensor([1, 2], dtype=torch.int64),
)
x = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
loss = op(x).square().mean()
loss.backward()
torch.optim.SGD(op.parameters(), lr=1e-3).step()
```

## Optional CUDA indexed-mechanism Torch adapter

The native `cellerator.prepare_mechanism` path and compiled Torch
`MechanismModule` require a CUDA indexed-mechanism build. `MechanismModule`
accepts CUDA FP32 coefficients with a declared `MechanismSpec`, batch capacity,
live-forward capacity, and `f32` or `mixed_f16` operation precision. It owns or
shares one prepared native handle and uses the guarded update/checkpoint
lifecycle. This is a specific mechanism contract, not a general Torch layer
system. A CPU-only build can use Product2 without importing these optional
symbols.

Native and compiled adapters are off in ordinary C++ builds. A CUDA/Torch
extension build must enable CUDA and Torch and use a compatible Torch/CUDA
toolchain. The Python package declares its optional Torch extra; one supported
build route is:

```sh
python -m pip install '.[torch]' \
  --config-settings=cmake.define.CELLERATOR_ENABLE_CUDA=ON \
  --config-settings=cmake.define.CELLERATOR_ENABLE_TORCH=ON
```

For direct CMake builds, enable `CELLERATOR_ENABLE_PYTHON=ON`,
`CELLERATOR_ENABLE_CUDA=ON`, and `CELLERATOR_ENABLE_TORCH=ON`. The compiled
Torch mechanism adapter depends on the native CUDA mechanism owner. Confirm the
configure summary and resulting targets; do not infer availability from
installed Torch alone.

## Sparse support and model slices

The sparse-support Torch package provides explicit support, indexing, routing,
gating, and selected model-slice interfaces, including `RankedGeneSlice` and
`KaryoplastEventSlice`. These consumers use controlled fixtures; their API
presence does not establish scientific validation or general support for
arbitrary single-cell inputs. Native sparse-support contracts can be used
without linking Torch. Consult the [sparse support foundation](../sparse_support_foundation.qmd)
and its linked evidence for exact guarantees.

An opt-in prepared entity-to-mechanism route query is exposed through the
sparse-support Torch binding and `cellerator.torch.prepared_routing`. It
consumes the caller's existing discrete route table and keys the prepared
projection with owner, relation, structure, route, and binding generations.
Its CPU adapter contract tests pass. The frozen binary also passed the 16-test
suite, exact CUDA correctness, and memcheck, racecheck, and synccheck. Bounded
V100 measurements found a long-reuse improvement for synthetic ranked `[32,8]`
routes, an inconclusive event result, and faster dense Torch results at larger
stress shapes. The current Python route query remains the default; prepared
routing is opt-in and is not a biological route learner. See the
[source-bound qualification report](https://github.com/tumlinso/Cellerator/blob/94c534fc794fe8cc3e0434e4a50f379401ddbb3c/bench/sparse_support/experiments/prepared_routing/REPORT.md)
for exact performance and qualification scope.

## Tokenization and resident CUDA

Tokenization is an optional ATen facility, separately selected with
`CELLERATOR_ENABLE_TOKENIZATION=ON`. It provides caller-driven tensor and
sequence operations; it does not define downstream event interpretation or a
general model loader. It can be CPU-only and does not require the CUDA
mechanism Torch adapter. The native `GeneTokenizer` base, concrete ranked-gene
and event-stream classes, and annotation-built `GeneVocabulary` are exposed
through thin bindings. Their execution inputs remain separately typed. See
[tokenization](../tokenization.qmd#gene-tokenizer-design).

Resident CUDA buffers and numeric operations have a distinct native owner and
optional thin Torch view. Their contracts cover stream, buffer lifetime,
borrowed-storage, and synchronization obligations. They are forward-only and
do not imply a general Torch autograd API. Use the facility-specific docs and
tests rather than treating all `cellerator.torch` submodules as one uniform
backend.

## Build a host Python extension directly

```sh
cmake -S . -B build-cpu \
  -DCELLERATOR_ENABLE_CUDA=OFF \
  -DCELLERATOR_ENABLE_PYTHON=ON \
  -DCELLERATOR_ENABLE_TORCH=OFF
cmake --build build-cpu --target cellerator_python_extensions
```

The installed wheel is the supported way to locate extension artifacts. Avoid
development-tree loader variables to make an installed package import. For
source and target ownership see [developer reference](../developer_reference.qmd);
for maturity and evidence see [current implementation](../current_implementation.qmd).
