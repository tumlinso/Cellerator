# Cellerator component ownership

The native [single-cell components](single-cell-methods.md) own preprocessing,
RNA velocity, and routed neighbor search. They use Cellerator numerical
primitives and views over caller-owned storage. AnnData++ owns arrays and file
I/O. The optional [AnnData component](anndata.md) reads backed selections into
native Cellerator buffers.

The original tracked methods and umbrella are preserved on the
[CCell archive branch](https://github.com/tumlinso/CCell/tree/codex/cellerator-native).

The former [AnnData++ and Velocity source archives](https://github.com/tumlinso/Cellerator/tree/d7414e7f830fcd25cac585399a0535b72e75ff01/quarantine)
remain available in Git history. Local cutover recovery files and the retired
Python AnnData source snapshot with its checksum manifest were backed up
outside the checkout.

The former `src/velocity` checkout was archived outside the repository with
its unpublished changes, Git metadata, and validation artifacts.

## Supported substrate surfaces

The native core, computational operands, generic feature bindings,
modeling, numerical operations, and selectable substrate SDK retain their
existing owners. Callers supply acquired numerical data and explicit identity,
layout, ownership, and execution contracts. File selection and scientific IO
belong to the consumer. Generic binary16 conversion remains in
`src/ops/numeric/binary16` and is independently buildable and installable as
`CelleratorBinary16`, with target `Cellerator::binary16`.

The Python AnnData adapter and its public API are retired. Generic NumPy-based
modeling and optional Torch, CUDA, and distributed adapters remain. Original
adapter, support-script, and affected test/example sources remain in the
backup described above.

## Native module hierarchy

Use `import cellerator;` for the common native module. It exports the existing
compressed, COO, and block-masked tensor interfaces from the `cellerator:tensor`
subdivision. The primary interface re-exports the partition with
`export import :tensor;`. Select `COMPONENTS module` and link
`Cellerator::cellerator` when using an installed package. Use the same
`cellerator::tensor` types through focused headers and `Cellerator::tensor`
when building without module tooling.

| Optional import | Source owner | Installed target |
|---|---|---|
| `cellerator.annotation` | `components/annotation` | `Cellerator::annotation` |
| `cellerator.tokenizers` | `components/tokenization` | `Cellerator::tokenization` |
| `cellerator.anndata` | `components/anndata` | `Cellerator::anndata` |
| `cellerator.nn` | `components/nn` | `Cellerator::nn` |
| `cellerator.preprocess` | `components/preprocess` | `Cellerator::preprocess` |
| `cellerator.velocity` | `components/velocity` | `Cellerator::velocity` |

CUDA routed search is available through the headers and `Cellerator::neighbors`
target supplied by `components/neighbors`.

Select components and import them explicitly. The primary module has the same
exports across builds. Use partitions to organize a named module when there
is a concrete need. Keep a source directory, a CMake target, and a C++
namespace independent of the named-module boundary. See the
[C++ module unit](https://eel.is/c++draft/module.unit) and
[import](https://eel.is/c++draft/module.import) rules.

Request `find_package(Cellerator CONFIG REQUIRED COMPONENTS annotation)` for
the installed Annotation component. Header consumers can link
`Cellerator::annotation` and include `<Cellerator/biology/gene_annotation.hh>`.
The annotation and catalog declarations remain in `cellerator::biology`.

### Root ownership

| Shared source | Supported use |
|---|---|
| Tensor views, numeric types, packing, layouts, operations, and derivatives | Native host and CUDA operation consumers. NN uses Product2 through its own target. |
| Axis and feature identity, sequence coordinates, and execution lifetimes | Shared data contracts across layouts and components. |
| Borrowed memory views and native parameter descriptors | Native single-cell methods, AnnData read bindings, Torch interop, and native training use the same contracts. |

A reusable numerical primitive or shared contract with a concrete supported
consumer earns a root source owner. Name that consumer and build target when
adding or moving a family. Put format readers, tokenizer policy, and framework
integration with the component that needs them. Annotation owns reference
annotation records, formats, and [optional compiled reference catalogs](../../components/annotation/catalogs/README.md).
Tokenization consumes gene catalogs to build vocabularies
with model token assignments. Keep modules as facades over
the same header declarations. Export low-level types when they are needed for
the public workflow and use focused headers for advanced controls.

Components use one build definition with in-tree targets or installed
dependencies. Implementations include the shared Cellerator headers they use
and link the owning targets. Cross-component calls use the dependency’s
public headers or module interface, with the dependency declared in CMake. Install their public headers and module sources through the
Cellerator package. The `packages/` directory contains pinned Git
submodules with their own repositories and independent builds. AnnData++
builds independently of Cellerator. The native method components build from
Cellerator-tracked source and export their targets through Cellerator.

## Retained inactive source

| Source | Current status |
|---|---|
| `components/CellShard` | Preserved in place; detached from supported Cellerator builds and package installs. No new CellShard development or IO transfer is included. |
| `include/Cellerator/interop/cellshard` | Preserved source only; excluded from native and SDK header installs. |
| `src/ops/velocity` and `tests/velocity` | Inactive native adapter and tests. The supported velocity method is owned by `components/velocity` and supplied by `Cellerator::velocity`. |
| `cmake/CelleratorVelocity.cmake` and `cmake/CelleratorVelocityModule.cmake` | Retained former integration definitions, no longer included or installed by supported builds. |
| `src/preprocess` | Dormant reference copy preserved in place; `components/preprocess` owns the active implementation. |
| `examples/torch` | Existing CellShard-oriented example sources retained without a supported Cellerator build option. |

Enabled retired CellShard options and required package components fail
explicitly. Their preserved sources can contain historical standalone build
files; those files do not re-establish supported Cellerator integration.
See the [CellShard server handoff](cellshard-server-handoff.md) for source
preservation, legacy data recovery, and the proposed packing work.

Source datasets, fixtures, recorded outputs, local histories, recovery trees,
and historical scientific receipts remain at their original paths. Historical
records retain their original dependency and qualification statements; they
are not evidence for the changed source. No dataset migration or deletion is
part of this ownership change.
