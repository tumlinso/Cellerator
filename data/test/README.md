Test data scaffold for local and CI-adjacent fixtures.

Layout:

- `reference/`
  - Place long-lived source fixtures here, such as a reference `.h5ad`.
- `generated/csh5/`
  - Persisted `.csh5` outputs derived from reference inputs or conversion tests.
- `generated/cache/`
  - Warmed CellShard cache trees, shard packs, and execution packs.
- `generated/logs/`
  - Debug logs, inspect dumps, and one-off validation artifacts.
- `generated/tmp/`
  - Scratch space for temporary conversion outputs and staging files.

Notes:

- The directory structure is tracked, but real fixture payloads are gitignored.
- Keep the canonical reference `.h5ad` name stable once chosen so tests and scripts can target it consistently.
- Generated outputs should be reproducible from the reference input or test setup.
