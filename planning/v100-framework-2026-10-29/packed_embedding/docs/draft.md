# Packed embedding bag on V100

## Current path and operation

The forward kernel assigns one CUDA lane to each `(segment, embedding_dimension)` output and serially accumulates that segment's sparse occurrences. Each occurrence maps a packed state logical index through a feature map and ragged feature codebook to an embedding row. An optional sorted exact-value sidecar brackets values between adjacent codebook levels and linearly interpolates their learned embedding vectors. Optional occurrence weights multiply each vector before the bag sum. Outputs use the table dtype with FP32 internal accumulation.

Backward assigns one thread to each occurrence, loops across embedding dimensions, computes one occurrence weight derivative if requested, and atomically accumulates FP32 table gradients. Codebook choices and exact state values are conditioning inputs and have no gradients.

## Risks and baseline

The tracked smoke test is only a host arithmetic sketch; its pointers to forward/backward do not invoke CUDA. There is no packed-embedding device oracle yet. Establish an independent CPU reference over sparse weighted occurrence bags, including repeated logical selections, empty segments, code bit widths, feature reordering, ragged codebooks, exact interpolation, FP16/FP32 tables and arbitrary upstream gradients.

The incumbent GPU path is the shipped kernel. Candidate comparison should report resident forward and backward CUDA-event times separately, plus an explicit decoded-dense gather/reduce baseline where useful. Include decode/setup when claiming a complete-path benefit. Use stable warmed repeats, matched outputs/gradients, and no overlapping GPU jobs.

## Candidate

After device correctness is established, compare current backward's one-thread-per-occurrence strided table accesses against a one-thread-per-(occurrence, embedding-dimension) variant with coalesced dimensions and the same FP32 atomic accumulation. Keep the incumbent unless the candidate is correct and wins consistently for a representative occurrence/dimension regime without a material regression in relevant small cases. Do not change the forward algorithm absent evidence.

## Validation

- Compile the smoke and library for SM70.
- Run the independent forward and backward oracle on an admitted V100; cover empty bags, duplicate occurrences, sorted overrides, bit widths and both table dtypes.
- Compare analytic gradients against an independently accumulated CPU formula and finite differences for occurrence weights.
- Run Compute Sanitizer memcheck, racecheck and synccheck on the admitted device smoke.
- Benchmark the incumbent and any candidate in a separate clean, exclusive, quiescent interval. Record source hashes, exact workload, device UUID, scheduler receipt, medians and spread.

No GPU device test, sanitizer result or performance claim is established by this draft. V100 timing, CUDA storage availability, and scheduler admission remain outstanding.
