# `compute`

Authoritative home for sparse ML math over CellShard matrices.

Rules:

- name code by math contract first: sparse linalg, reductions, transforms,
  selection, distances, gradients, neighbor scoring
- use NVIDIA library-backed paths as the default when cuSPARSE, cuBLAS, CUB, or
  NCCL matches the operation cleanly
- put measured or layout-specific hot paths under `custom/` folders beside the
  library path they replace
- keep CellShard data handling out of this tree; biology-facing preprocessing
  policy lives under `src/preprocess/`, while reusable preprocessing math lives
  under `src/compute/preprocess/`
- keep framework-independent sparse training code under `compute/sparse/ops`;
  libtorch-facing wrappers belong above that layer, not inside the runtime core
- keep reusable forward sparse projection contracts under `compute/sparse/project`
- keep reusable CUDA library handles and scratch ownership under `compute/runtime`;
  model code should acquire cuSPARSE/cuBLAS there instead of creating per-call handles
- document launch, HBM, PCIe, and synchronization costs at host boundaries

Primary folders:

- `core/`: small shared host/device utilities
- `runtime/`: pointer-first execution contexts, scratch arenas, and CUDA library caches
- `layouts/`: lightweight views over CellShard-owned layouts
- `sparse/`: reusable sparse math contracts and backend families
- `ml/`: ML-facing sparse math, sparse operators, and model-adjacent operators
- `neighbors/`: reusable neighbor scoring/search/top-k math plus forward-neighbor caller policy
