# `compute`

Authoritative home for sparse ML math over CellShard matrices.

Rules:

- name code by math contract first: sparse linalg, reductions, transforms,
  selection, distances, gradients, neighbor scoring
- use NVIDIA library-backed paths as the default when cuSPARSE, cuBLAS, CUB, or
  NCCL matches the operation cleanly
- put measured or layout-specific hot paths under `custom/` folders beside the
  library path they replace
- keep CellShard data handling, CellShardPreprocess biology policy, and
  CellShardNeighbors caller policy out of this tree
- keep framework-independent sparse training code under `compute/ml/autograd`;
  libtorch-facing wrappers belong above that layer, not inside the runtime core
- document launch, HBM, PCIe, and synchronization costs at host boundaries

Primary folders:

- `core/`: small shared host/device utilities
- `layouts/`: lightweight views over CellShard-owned layouts
- `sparse/`: reusable sparse math contracts and backend families
- `ml/`: ML-facing sparse math, autograd, and model-adjacent operators
- `neighbors/`: reusable neighbor scoring/search/top-k math only
