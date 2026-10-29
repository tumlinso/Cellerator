# Frozen serial posting kernels: q24 large-ranked profile

This run profiles one invocation each of `query_counts_kernel` and
`query_scatter_kernel` in the frozen pre-fastpath CUDA posting implementation.
The workload is the deterministic v3 large-ranked fixture (E=262,144,
N=4,096, B=32, 24 source terms per cell). The linked baseline executable and
the indexed source snapshot are immutable copies of the versions used for the
native baseline timings.

Unprofiled timing evidence remains the performance reference: q24 discovery
took 26.211 ms through serial postings and 3.172 ms through full rescan. NCU
replay values are diagnostic only, not application throughput.

## Collection

Nsight Compute 2025.2.1.0 was invoked explicitly at
`/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/profilers/12.9/Nsight_Compute/ncu` with
`--set full`, `--import-source yes`, and base clock policy. The two target
kernels each completed an 81-pass collection with one matching launch. Both
`.ncu-rep` files and command output are retained under `reports/`. The V100
does not support the PM-sampling timeline.

NCU warned that no source files were imported, despite the binary containing
CUDA line-info sections and the exact source snapshot being supplied through
`--source-folders`. Per-line source/stall attribution is therefore unresolved;
do not make source-line claims from this report until the path mapping issue is
repaired and collection is repeated. Helper extraction/diagnosis is pending.

## Identity and limits

- Frozen executable SHA-256: `a628a22579a576d51f942f91d3c0f279ec7c371fdec36f0b454a956e7a44ae7f`.
- Exact indexed source SHA-256: `a3aecefb1ed359342bbc6ffb6843bda079f8a28082030f07510a0e78275e3853`.
- Large-ranked fixture manifest SHA-256: `674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32`.
- Scheduler job `21971184-010c-43de-88bd-17f8708f7c88`, attempt
  `dc946c6c-f015-4098-abc5-ad026bfafb90`; succeeded, cleanup verified.
- This profiles only the serial baseline; it does not characterize the
  ordered-disjoint fastpath or imply biological validity of synthetic data.
