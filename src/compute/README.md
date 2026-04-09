# `compute`

Authoritative home for reusable computation in `Cellerator`.

Rules:

- put reusable kernels, operators, workspaces, and multi-GPU compute here
- keep dataset-specific CLI logic and one-off orchestration out of this tree
- prefer part/shard/fleet-scoped operators over monolithic end-to-end files
- treat `extern/CellShard` as the storage and staging substrate, not as the home for Cellerator compute
