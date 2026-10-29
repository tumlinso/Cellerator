# Publication checks, 2026-10-29

The user authorized publication after the implementation delivery. This addendum
records the subsequent checks; the original implementation receipts retain their
historical scope.

- Built the default CPU wheel with
  `/home/tumlinson/Software/miniconda3/bin/python -m pip wheel . --no-deps --no-build-isolation --wheel-dir /tmp/cellerator-wheel-20261029`.
  Installed it into an isolated `/tmp` target without changing the user environment.
- Against the installed wheel, 188 framework tests ran: 187 passed and the opt-in
  two-process Gloo test was skipped. Product2, sparse AnnData identity alignment,
  and the cellular interaction and trajectory training examples passed.
- Rebuilt the optional host tokenizer extension and gene vocabulary native test.
  The native test passed. The Python tokenizer suite ran 29 tests: 23 passed and
  six opt-in CUDA tests were skipped.
- Rendered all 36 Quarto pages successfully with
  `XDG_CACHE_HOME=/tmp/cellerator-publish-quarto-cache quarto render docs --to html --output-dir /tmp/cellerator-publish-docs-20261029`.
  Redirecting the XDG cache resolved the earlier cache database error without
  modifying the installed Quarto environment. Generated pages remain outside
  the source commit.
- Preserved the tracked `external/highway` and `external/htslib` submodule
  references. Their pre-existing local missing-directory state is excluded
  from publication.

The new CUDA kernels remain compiled but unexecuted on a GPU. This publication
adds no GPU numerical, sanitizer, performance, or biological qualification, and
does not claim the full legacy repository suite passed. Existing Project Control
claims were preserved after its front door returned `workflow_task_already_active`.
