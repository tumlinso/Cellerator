# `models`

Model workflows live here.

Current modules follow the existing `dataloader` / `model` / `train` /
`infer` split, with one umbrella header per workflow. Reusable compute belongs
under `src/compute/`; model wrappers and training orchestration belong here.

`developmental_time/` is now the native sparse scalar-time path. It accepts
Blocked-ELL and sliced-ELL batch views directly, uses Tensor Core-capable
Blocked-ELL `SpMM` for the sparse stem, and owns its train/eval/infer loop
without Torch or CSR batching.

`developmental_time_trajectory/` is the trajectory-aware sibling. It shares the
same native sparse stem but adds one external-graph aggregation stage plus
graph smoothness and directed-order penalties on scalar developmental time.

`state_reduce/` is the native CUDA identity-reduction path. It is explicitly
separate from the older Torch-bound `dense_reduce/` prototype and treats
Blocked-ELL / sliced-ELL as the intended steady-state sparse layouts.
