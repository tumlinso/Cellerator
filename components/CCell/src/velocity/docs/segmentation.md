# Current package boundary and historical source segmentation

Velocity lives under `components/CCell/src/velocity` as tracked source.
Consumers use `import ccell.velocity;` and link `ccell::velocity`. The
underlying `velocitypp` and `velocitypp.precision` modules retain their type
identity. The CPU reference uses caller-owned native views and FP32 or FP64
storage. For AnnData++ inputs, use the optional CCell adapter before binding
the reference.

## Historical relocation and qualification evidence

The [first-party relocation receipt](../planning/velocity-relocation.json) and
[portable segmentation receipt](../planning/portable-segmentation.json) record
an earlier source organization in the Cellerator checkout. Those JSON records
remain unchanged and describe their recorded paths and source snapshot. They do
not describe the current package boundary.

Earlier documentation and receipts describe an FP16 numeric-view migration,
Cellerator binary16 helper, and quarantined AVX2/CUDA implementations. They are
historical evidence for those older source snapshots only. They are not active
dependencies, targets, or implementation evidence for the CCell component.
See [validation](validation.md) for the preserved qualification status.

The original standalone source remains in `tumlinso/velocitypp`. Its imported
revision is recorded in the [CCell provenance guide](../../../README.md#source-and-provenance).
