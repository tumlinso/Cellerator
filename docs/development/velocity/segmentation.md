# Current package boundary and historical source segmentation

Velocity lives under `components/velocity` as tracked source. Consumers use
`import cellerator.velocity;` and link `Cellerator::velocity`. The CPU reference
uses caller-owned native views and supports binary16, FP32, and FP64 storage.
The optional AnnData component provides backed reads before binding the method.

## Historical relocation and qualification evidence

The [first-party relocation receipt](../../../components/velocity/planning/velocity-relocation.json) and
[portable segmentation receipt](../../../components/velocity/planning/portable-segmentation.json) record
an earlier source organization in the Cellerator checkout. Those JSON records
remain unchanged and describe their recorded paths and source snapshot. They do
not describe the current package boundary. The recorded accelerator files
can be found in the [archived Git tree](https://github.com/tumlinso/Cellerator/tree/d7414e7f830fcd25cac585399a0535b72e75ff01/quarantine/velocity-accelerators).

Earlier documentation and receipts describe an FP16 numeric-view migration,
Cellerator binary16 helper, and quarantined AVX2/CUDA implementations. They are
historical evidence for those older source snapshots only. They are not active
dependencies, targets, or implementation evidence for the native component.
See [validation](validation.md) for the preserved qualification status.

The original standalone source remains in [Velocity++](https://github.com/tumlinso/velocitypp/tree/570055b0f5ad8b491890ffcdf35aac74d3d541cb).
The tracked source used for the split is on the
[CCell archive branch](https://github.com/tumlinso/CCell/tree/codex/cellerator-native).
