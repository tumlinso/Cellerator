# `workbench`

Interactive and command-line workbench surfaces live here.

Current code includes the ncurses dataset workbench plus a shared runtime backend
under `src/workbench/runtime/`.

The intended split is:

- `src/apps/workbench/dataset_workbench_main.cc`: ncurses adapter surface
- `src/workbench/dataset_workbench.hh`: stable public facade for UI and Python callers
- `src/workbench/runtime/`: shared dataset workflow implementation, including summary,
  conversion, preprocess orchestration, metadata rewrite, finalize, and browse-cache
  helpers

`dataset_workbench.hh` is also the current library seam behind the optional
`cellerator` Python wrapper. The ncurses app and the Python package are both
expected to stay thin over the same planning, conversion, inspection, and
preprocess functions rather than forking separate workflow logic.
