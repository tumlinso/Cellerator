# `compute`

Authoritative home for reusable computation in `Cellerator`.

Rules:

- put reusable kernels, operators, workspaces, and multi-GPU compute here
- keep dataset-specific CLI logic and one-off orchestration out of this tree
- prefer part/shard/fleet-scoped operators over monolithic end-to-end files
- treat `extern/CellShard` as the storage and staging substrate, not as the home for Cellerator compute
- document launch and transfer costs at the host call site; comments should say whether a call is launch-bound, bandwidth-bound, synchronization-heavy, or amortized across a shard/part
- call out any hidden host round-trip such as `.to(torch::kCPU)`, `cudaDeviceSynchronize`, or blocking `cudaMemcpy`, because those often dominate small V100 workloads more than the math does
