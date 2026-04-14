# Docs Workspace

`docs/` now holds the active Quarto documentation source for the Cellerator pipeline.

Primary source files:

- `_quarto.yml`: Quarto book configuration
- `index.qmd`: docs entrypoint
- `architecture.qmd`: ownership split and module map
- `ingest_and_runtime.qmd`: bounded-memory ingest and owner-service runtime contract
- `compute_and_models.qmd`: reusable compute plus downstream trajectory and model layers
- `local_pipeline_ops.qmd`: local operational wrappers under `docs/pipeline/`
- `developer_reference.qmd`: build, test, and repo-map quick reference

Supporting material that may still live here:

- `pipeline/`: local-only workflow code such as FASTQ download helpers and unified preprocessing wrappers
- `public_omics/`: older research planning and shortlist artifacts
- `notes.txt`: optional background notes, not the primary docs source

Boundary rules:

- Nothing under `docs/` is required to configure, build, or test Cellerator.
- `docs/pipeline/` may call external tools or built Cellerator binaries, but product code must not depend on it.
- If a workflow here becomes reusable Cellerator functionality, migrate it into `src/` in a separate implementation pass.
