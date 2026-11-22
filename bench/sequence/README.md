# Exact sequence count benchmarks

Build `cellerator_sequence_bench` for the scalar CPU path or
`cellerator_sequence_cuda_bench` in a CUDA-enabled build. Each takes optional
positional arguments `bases warmup repeats` and checks a deterministic
16-base motif workload against a text scan. The defaults are 1,048,576 bases,
3 warmups, and 20 timed repetitions.

The JSON output reports packing and predicate preparation once, upload on
CUDA, and the repeated resident batch. The resident measurement includes
resetting the count for each operation. Fixture generation, oracle computation,
allocation, and output transfer are outside the timed interval. Run CUDA
benchmarks through the shared host's Device Scheduler.
