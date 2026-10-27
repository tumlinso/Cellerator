# Parallel workset compaction probe

This probe replaces the production one-thread-per-cell filter/merge with a
CTA-per-cell tiled-prefix and stable-rank scatter candidate. It is experiment
only; production source and the official benchmark driver are not modified.

The fixture builder is copied from the official CUDA indexing benchmark's
deterministic support/query/condition generator. The E=2176, B=64, skew=0.9,
gate=0.5, seed=20261027 fixture reproduces input digest
`0xf82d50df3b097939`. Host mode performs no CUDA calls and checks posting rows
plus a map-based independent workset oracle:

```sh
/tmp/cellerator_parallel_workset_build129/parallel_workset_probe \
  --host-only --label e2176_b64_skew09 --edges 2176 --cells 64 \
  --entities 256 --source-skew 0.9 --gate-density 0.5 --seed 20261027
```

Configure/build for Volta with CUDA 12.9 in a separate build directory:

```sh
cmake -S bench/sparse_support/experiments/parallel_workset \
  -B /tmp/cellerator_parallel_workset_build129 \
  -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCUDAToolkit_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9 \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CXX_COMPILER=/usr/bin/g++
cmake --build /tmp/cellerator_parallel_workset_build129 -j 4
```

The CUDA run first checks production and candidate outputs against the host
oracle, checks GPU postings against host posting rows, and checks primal and
derivative capacity overflow without partial public-output writes. Timed scopes
are paired in alternating order: frozen-candidate workset, full bitmap gate
plus workset, and GPU postings plus bitmap gate plus workset. The latter is the
resident total indexed pipeline; cold host index build/projection transfer are
outside the timing window and remain separate setup costs. Each case reports
21 event-timed samples with three warmups and three repeated iterations.

The finite scheduler request and matrix runner are retained beside this file.
Use Device Scheduler with the Cellerator client config and the exact `resource_id`
in that request; no direct CUDA launch is an authorized measurement path.
