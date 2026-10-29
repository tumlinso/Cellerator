# CelleraTorch history

This archive contains two different historical threads. `AGENTS.md` and the original mechanism notes preserve records for the former component direction. `paused_runtime_workspace.md` and `paused_runtime_design.md` preserve a later documentation-only proposal to explore a simplified Torch-facing runtime.

The paused proposal did not implement a package, build target, or runtime and is not the current product plan. Cellerator remains a PyTorch extension: it owns reusable native operations and structure-aware execution; Torch supplies its general framework facilities. The current direction is in [architecture](../../architecture.qmd), [scope](../../scope.qmd), and [Python bindings](../../development/python-bindings.md). The two proposal files are preserved without rewriting their original text.
