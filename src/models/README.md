# `models`

Header-first libtorch workflows live here.

Current modules follow the existing `dataloader` / `model` / `train` /
`infer` split, with one umbrella header per workflow. Reusable compute belongs
under `src/compute/`; model wrappers and training orchestration belong here.

`developmental_time/` is the framework-managed baseline. The separate
`developmental_time_cuda/` sibling is the benchmarkable CUDA-native path for
the same scalar-time plus auxiliary-bin objective.
