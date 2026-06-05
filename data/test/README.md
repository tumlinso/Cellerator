Cellerator-local test data scaffold for local and CI-adjacent fixtures shared
across the umbrella workspace. Deprecated CellStack wrapper checkouts should not
keep a separate root-level `data/` tree; local fixture payloads live here.

Layout:

- `reference/`
  - Place long-lived source fixtures here, such as a reference `.h5ad`.
- `generated/csh5/`
  - Persisted `.csh5` outputs derived from reference inputs or conversion tests.
- `generated/cache/`
  - Warmed CellShard cache trees and CSPACK files.
- `generated/blocked_ell_optimization/`
  - Local outputs used while tuning or validating blocked ELL kernels.
- `generated/logs/`
  - Debug logs, inspect dumps, and one-off validation artifacts.
- `generated/tmp/`
  - Scratch space for temporary conversion outputs and staging files.

The directory structure is tracked, but real fixture payloads are gitignored.
This includes large `.h5ad`, `.csh5`, `.mtx`, `.npy`, sequence alignment, and
FASTQ/FASTA files. Keep canonical reference fixture names stable once chosen so
tests and scripts can target them consistently.
