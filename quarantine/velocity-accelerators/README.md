# Quarantined velocity accelerators

This directory preserves the CUDA and AVX2/F16C work excluded from the portable
velocity component. It is outside every active source list, package export,
test registration, and dependency-discovery path. There is no build entrypoint
here and no active option to enable these implementations.

- `cuda/` holds the CUDA module, implementation, private bridges, tests,
  installed-consumer sources, benchmarks, Scheduler helpers, and CUDA documents.
  Original shared numerical headers are preserved beside the CUDA sources.
- `avx2/` holds the original mixed scalar/AVX2 translation unit, its interface,
  and its original test source. Portable bodies remain in the active component;
  the archived copy retains the original fallback dependencies and ISA branches.
- `support/` preserves original mixed build/package files, interfaces,
  qualification drivers, comparison tools, and compact historical evidence.
  Historical receipt files also remain at their original locations.

Paths within each section mirror their original repository-relative paths.
`root-segmentation-files.json` records root-owned moves and snapshots; the
[completed segmentation inventory](../../src/velocity/planning/portable-segmentation.json)
records the component paths moved, snapshotted, or edited. Recorded hashes
describe preserved bytes, not a new numerical qualification.

The archived code and tools retain original relative-path assumptions. They
are source and evidence for review or deliberate recovery, not a maintained
standalone accelerator package. In particular, a mixed-script invocation may
require its original source/build layout and ignored raw artifacts.

The preceding V100 work established backend numerical agreement on the recorded
fixtures and passed the recorded CUDA sanitizer runs. Its bounded sparse-workload
measurements demonstrated no end-to-end CUDA gain and no reliable AVX2 gain.
The CUDA execution decomposition was underparallelized. Those historical results
do not qualify this segmentation or promise performance on another workload.

FP16 storage and file encodings belong to the external data package. They have
not been moved here. The active velocity component retains portable binary16
conversion, FP32/FP64 arithmetic, mixed-storage inputs, and explicit precision
utilities. It contains no F16C requirement.

The archive now lives at `Cellerator/quarantine/velocity-accelerators/`; the
portable CPU component lives at `Cellerator/src/velocity/`. The archive remains
outside the active build and package exports. No builds, tests, benchmarks, or
GPU jobs were run for this relocation. Historical numerical and performance
results above remain tied to their original source snapshots and conditions.
