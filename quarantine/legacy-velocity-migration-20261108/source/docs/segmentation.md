# Portable source boundary

This repository contains the independently buildable native reference
component. Its active methods are CPU implementations. There is no selectable
`velocity::cpu` namespace, runtime backend switch, AVX2/F16C implementation,
CUDA implementation, toolkit discovery, ISA option, intrinsic header, or
accelerator package export. The compiled C++ functions are the portable
reference implementation; “CPU-only” describes the available execution
target, not a selectable backend.

The parent Cellerator checkout preserves the former CUDA, AVX2/F16C, mixed
comparison, and accelerator support sources under
`quarantine/velocity-accelerators/`. That archive is provenance for earlier
work and is outside this library's source list, install tree, and current
qualification. Do not infer an active capability from those files or receipts.

## Active component

The library consists of the named public modules and their compiled portable
implementations under `src/`, plus its package files and the independent
AnnData++ and `Cellerator::binary16` dependencies. The main public modules are
`velocity` and `velocity.precision`; the standalone target is
`velocity::velocity`. The documented supported toolchain is Clang 18 with its
matching `clang-scan-deps` and the Ninja generator for C++20 module dependency
scanning. CMake enforces Clang 18 or newer and Ninja; that acceptance check is
not a claim that newer compiler versions have been qualified.

The numerical boundary is `NumericView`/`NumericEdit` acquisition and the
plain descriptors used by block operations. Metadata for rank one and two is
inline. Dense affine views use `.subview(axis,start,count,step)`; sparse
subviews are unsupported. Irregular gene selections are retained by
`Input::genes` and compiled into the prepared order. Borrowed views rely on
caller-owned storage; retained views keep an explicit shared owner. Acquired
`NumericBytes`, `NumericDense`, `NumericBlock`, and `DenseOutput` descriptors do
not own their data.

Callers validate the fixed run once, schedule independent `GeneRange`s, and
provide each worker its own workspace and callback/provider state. Initialization
and refinement have an explicit join boundary around global selection. The
library has no scheduler or locks. Direct output acquisition and block writes
avoid a sink conversion buffer; matching-type moment fields can be exposed as
borrowed workspace views.

The serial workspace cap defaults to 64 MiB and includes worker numerical and
mask scratch plus requested selection and sink-conversion scratch. Promotion
checks a separate 64 MiB default cap against workspace scratch plus declared
provider staging. Parallel callers account for the sum of worker scratch,
global selection scratch, and provider staging. Input, retained result arrays,
and caller-owned dense outputs remain explicit separate storage.

AnnData++ is independent and required for its array, axis, and dataset
interfaces. It does not provide this FP16 view contract. `Cellerator::binary16`
is a small CPU-only, header-only helper; it does not imply a dependency on the
full Cellerator runtime.

## Historical relocation record

The [relocation inventory](../planning/velocity-relocation.json) and
[segmentation inventory](../planning/portable-segmentation.json) preserve
the earlier file movement and quarantine decisions. At that point the
relocation receipt explicitly reported no configure, build, test, benchmark,
or GPU run. Later FP16 ownership checks and earlier Milestone 1–3 records have
their own source identities in [validation](validation.md); none qualifies
the current caller-scheduled API. Historical JSON and quarantined bytes remain
unchanged.
