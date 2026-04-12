# `models`

Header-first libtorch workflows live here.

Current modules follow the existing `dataloader` / `model` / `train` /
`infer` split, with one umbrella header per workflow. Reusable compute belongs
under `src/compute/`; model wrappers and training orchestration belong here.
