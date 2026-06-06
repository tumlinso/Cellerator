# Pipeline Operations

`docs/pipeline` is the local-only operational workflow surface around the Cellerator pipeline. It contains helper code that is useful for dataset acquisition and preprocessing, but it is not packaged as part of the core Cellerator product surface.

This area is intended for:

- accession and FASTQ download helpers
- dataset manifests and local environment configuration
- unified preprocessing wrappers, including NVIDIA Parabricks-driven steps when appropriate
- run logs and provenance helpers for local pipeline runs

This area is not intended for:

- CMake targets, install rules, or exported libraries
- required runtime dependencies of `src/`
- long-term reusable product code that belongs in Cellerator proper

## Layout

- `download/`: fetch helpers for SRA, ENA, GEO, or direct FASTQ URLs
- `preprocess/`: unified wrappers that turn raw inputs into paper-ready intermediate artifacts
- `config/`: local env templates plus download and preprocess manifests

## Quick Start

1. Copy `config/local.env.example` to a local env file and set absolute storage and tool paths.
2. Fill `config/download_manifest.example.tsv` or create a sibling manifest with real accessions and output roots.
3. Run `download/fetch_fastq.sh --config <env-file> <manifest>` to preview the download plan.
4. Re-run with `--run` to execute the selected downloads.
5. Fill `config/preprocess_manifest.example.tsv` with dataset-specific commands.
6. Run `preprocess/run_unified_preprocess.sh --config <env-file> <manifest>` to preview or execute the preprocessing plan.

Both entrypoints default to dry-run mode so the command plan is visible before local state changes.

## Current Env Surface

The current scripts still use legacy `CELLERATOR_PAPER_*` and `PAPER_*` environment-variable names. Those names are retained for compatibility with the existing shell wrappers even though this directory is now documented as pipeline operations rather than paper scaffolding.

## Contract

- Inputs come from local manifests, dataset roots, tool paths, and reference files.
- Outputs should land under user-managed local roots rather than the tracked source tree.
- If a downstream step needs Cellerator, call built binaries or import emitted artifacts through normal ingest paths instead of wiring `docs/pipeline` into the product build.
