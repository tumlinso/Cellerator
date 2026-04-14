# `workbench`

Interactive and command-line workbench surfaces live here.

Current code includes the ncurses dataset workbench and its supporting ingest
inspection and preprocess orchestration helpers.

`dataset_workbench.hh` is also the current library seam behind the optional
`cellerator` Python wrapper. The ncurses app and the Python package are both
expected to stay thin over the same planning, conversion, inspection, and
preprocess functions rather than forking separate workflow logic.
