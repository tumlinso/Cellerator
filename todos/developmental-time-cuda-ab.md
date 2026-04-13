# Current Objective

## Summary
Implement a matched-output libtorch baseline and separate CUDA-native developmental-time module, plus a fair A/B benchmark harness for train and inference.

## Planning Notes
- The libtorch baseline must keep the current module path while switching to the matched scalar-time plus auxiliary-bin contract.
- The CUDA sibling must stay separate from the baseline so both implementations remain buildable and benchmarkable in the same tree.

## Assumptions
- Volta sm_70 is the target.
- The current developmental_time code can be adapted in place into the matched-output libtorch baseline.
- v1 benchmark data is synthetic with a real-data hook under the same contract.

## Suggested Skills
- `cuda-v100` - Own the sparse CUDA boundary and topology-aware multi-GPU path.
- `compare-benchmarks` - Create the fair A/B harness and benchmark contract.

## Useful Reference Files
- `src/models/developmental_time/dT_model.hh` - Current baseline surface to adapt.
- `src/compute/autograd/autograd.hh` - Sparse CUDA building-block surface for the new module.
- `bench/benchmark_mutex.hh` - Mandatory benchmark serialization primitive.

## Plan
- Adapt the libtorch developmental_time module to the matched-output scalar-plus-bin contract.
- Add a separate developmental_time_cuda module with CUDA-owned forward/backward/loss/update hot path.
- Add shared tests and A/B benchmark targets with stable phase accounting.

## Tasks
- [x] Adapt libtorch developmental_time baseline
- [x] Add developmental_time_cuda module
- [x] Add shared compile/runtime tests
- [x] Add A/B benchmark harness and targets

## Blockers
_None recorded yet._

## Progress Notes
- Adapted `src/models/developmental_time/` to emit scalar `predicted_time` plus auxiliary `time_bin_logits`, and switched the training loss to SmoothL1 plus cross-entropy on fixed equal-width bins.
- Kept the sampler's exact-label balancing path but changed emitted bucket labels to fixed equal-width time bins so both implementations share one supervision contract.
- Added `src/models/developmental_time_cuda/` as a separate CUDA-native sibling with a custom sparse stem built from `compute/autograd` CSR `SpMM` patterns and a manual SGD update path.
- Added `developmentalTimeCudaRuntimeTest` and `developmentalTimeABBench` targets, plus wrapper scripts for `compare-benchmarks`.
- Verified `developmentalTimeCompileTest`, `developmentalTimeCudaRuntimeTest`, `modelCustomOpsTest`, direct benchmark runs, and an end-to-end compare harness run.

## Next Actions
- Run larger and real-data-hook benchmark scenarios on the intended V100 host and decide whether the CUDA path needs the planned pair-local multi-GPU extension next.

## Done Criteria
- Both implementations build together, expose the same outputs, and benchmark under one fair contract.
