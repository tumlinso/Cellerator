# Former AnnData++ extensions

This quarantine preserves the unpublished FP16 storage and CUDA work removed from AnnData++. It is not an active package, build target, or CellShard format implementation.

`source/` holds exact pre-migration source, tests, documentation and historical receipts. `patches/` holds the original tracked changes against the published rename base. `manifest.json` records their original names and SHA-256 hashes. Historical paths, commands and receipts are intentionally unchanged.

Portable binary16 load and round operations are moving to Cellerator numeric support; velocity retains FP16 through its own explicit storage views. The archived storage codecs and CUDA sources remain excluded from builds, installation and active tooling. No GPU execution is performed by this migration.
