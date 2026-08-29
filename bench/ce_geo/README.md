# CE-GEO benchmark and evidence harness

This package defines decision-quality CE-GEO benchmark methodology. It does
not acquire a GPU, create a lease, arm a controller, run a benchmark, or claim
that a candidate is promoted. Hardware execution remains forbidden until the
configured controller supplies the required benchmark and GPU leases.

## Evidence boundary

Every campaign starts from a committed command manifest derived from
`evidence/schema/command_manifest.example.json`. Commands are argv arrays, not
shell strings. The manifest freezes build, correctness, and measurement
commands; warmup and repeat counts; maximum accepted median-absolute-deviation
spread; resource requirements; provenance capture; and the complete cold and
warm phase sets.

Measured evidence is valid only when it records:

- a clean immutable source revision, submodule revisions, todo revision, and
  working-tree status digest;
- device UUID/name/PCI identity/performance class/driver and a checksum-pinned
  topology capture;
- C++ compiler, CUDA toolkit, NVCC, CMake, build mode, architecture, CMake
  cache digest, and benchmark binary digest;
- exact argv, working directory, and a redacted environment digest;
- controller lease identifiers plus acquisition of the repository benchmark
  mutex from `bench/benchmark_mutex.hh`;
- correctness and numerical referee evidence before timing;
- raw repeated samples for every required cold and warm phase;
- consumer-visible complete samples, their recomputed median and MAD percent,
  and an explicit contamination disposition.

Do not place credentials, tokens, complete environments, hostnames, user
identities, or unredacted process listings in committed evidence. Record stable
digests and bounded hardware/toolchain facts instead.

## Complete-cost accounting

Cold phases are host preparation, semantic packing, projection construction,
backend preparation, static value packing, and persistent upload. Warm phases
are dynamic H2D movement, dynamic input packing, kernel execution, exact residual execution,
epilogue, order transformation, synchronization, communication, D2H evidence
movement, and consumer-visible completion. A zero-duration phase must still be
recorded; omission is not equivalent to zero.

Record raw cold costs before amortization. Any amortized result must retain the
exact structure, projection, prepared-program, value-generation, dense-layout,
work-window, and graph-replay reuse counts that justify it. Persistent upload
is not charged per use unless it actually occurs per use. Kernel-only timing is
never a complete result.

Use at least one warmup and an odd number of at least five measured repeats.
The command manifest fixes the exact counts before execution. The validator
recomputes the complete-sample median and MAD percent. Evidence exceeding the
declared spread is rejected unless marked contaminated, and contaminated
evidence can never be accepted for planner selection. After three contaminated
attempts, report an inconclusive result; do not choose a winner by noise.

## Controller procedure

The external controller must perform these steps in order:

1. create a clean immutable snapshot and capture its source identity;
2. build and run independent correctness and numerical checks;
3. prove quiescence and acquire benchmark, GPU, and any profiler leases;
4. acquire the repository benchmark mutex and check device-memory headroom;
5. capture device, topology, clocks, thermal/power/ECC state, toolchain, build,
   command, and redacted environment identities;
6. execute the frozen warmups and repeats while collecting every phase;
7. record contamination observations and release all resources;
8. validate the resulting JSON before it can become campaign evidence.

Profiling requires a separate profiler lease and explains an already-correct
result; profiler counters alone never promote a candidate. V100 `sm_70`
remains the current baseline. Results from another architecture use a distinct
performance class and cannot silently replace it.

## CPU-only validation

From the repository root:

```bash
python3 bench/ce_geo/harness/validate_evidence.py \
  --manifest bench/ce_geo/evidence/schema/command_manifest.example.json \
  --json

python3 -m unittest discover \
  -s bench/ce_geo/harness \
  -p 'test_*.py'
```

Supplying `--evidence PATH` additionally validates a result record and checks
that its `command_manifest_sha256` matches the exact manifest bytes. Both
commands are CPU-only schema and methodology checks and report
`performance_run=false`.
