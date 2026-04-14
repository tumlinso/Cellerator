# Pipeline Config

This directory holds local configuration templates for the local pipeline wrappers under `docs/pipeline/`.

- `local.env.example`: environment variables for data roots, log roots, and tool overrides
- `download_manifest.example.tsv`: sample manifest for FASTQ acquisition
- `preprocess_manifest.example.tsv`: sample manifest for unified preprocessing runs

The example files are templates only. Copy them to local working names before filling in real paths, accessions, and references.

The shell wrappers currently expect legacy `CELLERATOR_PAPER_*` and `PAPER_*` variable names. The docs reframe this as pipeline configuration, but the operational interface is unchanged.
