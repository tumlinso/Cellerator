# Native C++ biological modeling foundation — 2026-10-31

The delivered surface is an optional native C++20 modeling library backed by
ATen/autograd and ordinary Torch optimizers. Models, inputs, results, selectors,
and readouts are concrete callable types. Numeric parameter groups have an
explicit owner; authoring requires neither a model superclass nor member/module
registration. Biological bindings, prepared relations/targets, queries, explicit
conditioning, supervision, overlapping programs, generic cells, and optional
manifest/checkpoint IO accompany the neural components.

The [native guide](../../docs/native_nn.qmd) documents the public interface.
[Reference models](../../examples/native_nn/reference_models.hh) and
[training loops](../../examples/native_nn/train.cc) cover expression and
repeated event/history inputs, shared base computation, and auxiliary ATAC heads
with frozen and jointly trained base parameters. Dynamics retains its explicit
`DynamicalModel`/`DynamicalCell` route; ordinary encoders need no transition rule.

## CPU, packaging, and language boundaries

- Native CPU CTest: **15/15 passed**, with parameters/layers/encoders,
  identity alignment, relation attention, query/readout ownership, generic Cell
  integration, supervision, interaction programs, IO, and both model families.
- The reference CPU training executable ran through expression MSE, event
  teacher-forced cross-entropy, and frozen/joint auxiliary BCE. Snapshot tests
  establish parameter-update behavior; these synthetic examples are not
  biological evidence or a convergence guarantee for all objectives.
- Python integration: **337 run, 323 passed, 14 explicit skips**, no errors or
  failures. [Machine-readable results](python-validation.json) identify the
  extension and source hashes and enumerate skips. The focused shared-native
  bridge passed **9/9**, including a real Python biological manifest round-trip
  through the native JSON implementation ([record](bridge-validation.json)).
- A native-enabled wheel was built without changing the installed package.
  A fresh extracted-wheel subprocess checked Torch-free root/biological imports,
  generic cells, native encoding/attention gradients, Product2, and manifest
  conversion. All 82 packaged Python source files matched the checkout
  ([wheel record](wheel-validation.json)). This is a CPU native-extension wheel;
  it does not claim CUDA execution or an installed C++ SDK.
- Standalone installed `find_package(Cellerator COMPONENTS nn)` consumers built
  and executed ordinary autograd/optimizer composition. Optional tokenization
  vocabulary adaptation also compiled and passed. Generic biological bindings
  and cell handles have Torch-free header routes.
- AddressSanitizer regressions cover borrowed base-result and selector values,
  rvalue model/input/conditioned invocation storage, and custom full-head
  results whose gathers return spans. A single typed batch frame retains these
  values; its predictions share one lifetime token. Borrowed lvalue dependencies
  remain caller-owned. [Reproducers, original finding, and runner](lifetime/README.md).

Detailed host commands and provenance are in [build](build/README.md) and
[final validation artifacts](final/). The earlier validation snapshot in
`final/pre-root-query-view-validation.json` precedes the last full-head-view
fix; the separate root final records identify the post-fix state.
The [final host record](final/root-final-validation.json) verifies the final
header hash in both installed prefixes, CPU tests, ASan regressions, and both
module consumers. The full **38-page** documentation book rendered in an
isolated temporary directory; the sibling-source link was separately rechecked
([render record](final/docs-render.json)). No rendered output or Quarto cache
was written into the repository.

## Optional host modules

The Clang 18/CMake 3.28/Ninja consumer imports `cellerator.nn`, trains a native
model, shares ATen/Parameters types with a conventional header translation
unit, and includes a header-free importing translation unit. It passed its
single CTest. A separate NVCC 12.9/GNU 12 ordinary-header CUDA bridge compiled
and linked against the CUDA-enabled SDK and ran the same CPU consumer. NVCC
never parsed module syntax and no device work occurred in these checks.
[Qualification and reproduction](modules/qualification.md).

The qualified configuration supports several importing translation units in
one executable. Separate executable targets importing the same installed module
triggered a CMake 3.28/Ninja duplicate-rule limitation; that project shape is
not qualified. Headers remain independently usable.

## Admitted V100 qualification

Existing numerical kernels were reused. Native adapters are first-order,
respect explicit FP16/FP32 storage and FP32 accumulation, use the current
CUDA stream, retain per-invocation workspaces/foreign owners, and perform
saved-variable mutation checks. No new numerical kernel was introduced.

- Normal adapter qualification: **11 groups passed, zero skips**.
- Compute Sanitizer adapter qualification: **9 groups passed, two explicit
  instrumentation-incompatible host-gated fixture skips; zero memory errors**.
  Those two fixtures passed the required normal run.
- High-level CUDA composition: **three checks passed** in both normal and
  Compute Sanitizer runs, **zero skips and zero memory errors**. These cover
  prepared attention values/VJPs, frozen shared biological neural composition,
  and overlapping programs.

All device runs used Scheduler-admitted Tesla V100/SM70 execution with verified
cleanup. Exact sources, executable hashes, jobs, and unsuccessful attempts are
retained in the [adapter report](derivatives/REPORT.md) and
[composition report](derivatives/COMPOSITION_REPORT.md). Later host-only query
lifetime, checkpoint-group, and install/config changes are tested separately;
they do not retroactively qualify a different CUDA executable. No performance
or biological-validity claim follows.

## Explicit limits and preservation

Native CUDA differentiation is first-order only; explicit kernel selection
rejects active dropout and unsupported conditions. Arbitrary globally installed
pluggable CUDA allocators, CUDA graph capture, and broad precision/toolchain
portability are outside this qualification. The default native allocator was
exercised; cudaMallocAsync support and DLPack-specific round trips were not
independently qualified. See the adapter report for exact lifetime boundaries.

The first NN backend intentionally exposes ATen tensors. This does not make
ATen the universal representation of Cellerator data. Low-level caller-owned
storage APIs and existing guarded native-owner checkpoint contracts remain.
Distributed scheduling, additional backends, complete Karyoplast/Cytoplast
architectures, Hub publication, and scientific identification claims remain
outside this milestone.

This task created no commit or push. The shared workspace received external
commits during implementation; initial preservation and evidence snapshots
retain their actual identities. No reset, cleanup, or source restoration was
performed. Project Control returned an already-active-task attention response
without a workflow handle; no workflow completion was asserted.
