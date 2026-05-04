# `compute/sparse`

Reusable sparse math contracts over CellShard matrix views.

Each operation family should prefer a library-backed implementation first and
place layout-specific or fused hot paths under a sibling `custom/` folder.
Public callers should choose an operation contract, not a backend filename.

`project/` owns forward sparse-matrix times dense-weight projection contracts
for CSR, Blocked-ELL, and quantized Blocked-ELL layouts. Backward and optimizer
support stays under `compute/sparse/ops`.
