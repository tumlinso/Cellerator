# Installed substrate component gate

From the designated merged CE lineage:

```
python3 -B tests/substrate/install/check.py --build-dir /tmp/ce-is1-build-host --prefix /tmp/ce-is1-sdk-a
```

Baseline installs actual native host contracts, host relation/local arithmetic,
fixed stage program, ordered incidence, Cellpack geometry, native effects, and
host sequence operations. The package exports `Cellerator::effects` and
`Cellerator::sequence_ops` from their native source owners. Requests for the
retired Moonshot components fail with an explicit diagnostic.

The gate copies the installed SDK to a temporary independent prefix and compiles
consumer sources using only installed targets/public headers. It checks exported
capability metadata and dependency direction. Independent effects and sequence
imports exercise affine composition, packing, and motif counting without
loading unrelated numerical, CUDA, or Torch targets.

After MERGE-A:

```
python3 -B tests/substrate/install/check.py --build-dir /tmp/ce-is1-build-integrated --prefix /tmp/ce-is1-sdk-a --require-integrated
```

This mode requires all accepted source headers and compiles the actual installed
vertical path: structured state → cold identity packing/native relation → patch →
private transport → affine effect → repeated prepared native sweep. A missing leaf
fails configuration. There is no fallback to a sealed seed or another worktree.
Baseline BUILD evidence does not establish that the integrated mode has passed.

Producer roots are independently selectable with
`-DCELLERATOR_SUBSTRATE_COMPONENTS=sequence_ops` (semicolon-separated names).
Component dependency closure is explicit. Installed package imports load component
exports lazily. Header presence alone does not imply all declared native GPU or
adapter symbols are linked. `Cellerator::indexed_incidence` supports incidence only; the original full
`Cellerator::indexed_mechanism` is unavailable in this host SDK.
Torch adapters and compiler SDK packaging retain their existing separate builds.

Optional SM70 device-linear build for qualification through Device Scheduler:

```
python3 -B tests/substrate/install/check.py --build-dir /tmp/ce-is1-build-sm70 --prefix /tmp/ce-is1-sdk-sm70 --enable-sm70 --cuda-compiler /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc --cuda-host-compiler /usr/bin/g++-12
```

This builds the real provider but does not launch a GPU correctness fixture. CUDA
12.x and architecture 70 are required. Host differential/prepared components require
CUDA SDK headers to compile their existing public header, without a GPU launch.

Shared root changes are specified in `cmake/substrate/ROOT_INTEGRATION.md`.

The [original dependency manifest](https://github.com/tumlinso/Cellerator/blob/2c2c222f36c3aa1f7f8bb22d97031002c6420978/tests/substrate/install/dependencies.sha256)
records hashes for the historical SDK and its experiment inputs.
