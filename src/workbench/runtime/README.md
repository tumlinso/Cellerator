# `workbench/runtime`

Shared dataset workflow backend for the workbench facade.

This area is not the ncurses UI. It owns the implementation behind
`src/workbench/dataset_workbench.hh`, including:

- dataset summary and metadata inspection
- dataset conversion orchestration
- preprocess analysis, persistence, and finalize flows
- browse-cache generation and rebuild helpers

`src/workbench/` should stay adapter-facing. Reusable low-level math still
belongs in `src/compute/`, especially `src/compute/preprocess/`.
