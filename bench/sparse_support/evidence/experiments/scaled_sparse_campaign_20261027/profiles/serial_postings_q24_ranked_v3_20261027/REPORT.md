# Frozen serial posting kernels: q24 large-ranked profile

This run profiles one invocation each of `query_counts_kernel` and
`query_scatter_kernel` in the frozen pre-fastpath CUDA posting implementation.
The workload is the deterministic v3 large-ranked fixture (E=262,144,
N=4,096, B=32, 24 source terms per cell). The linked baseline executable and
the indexed source snapshot are immutable copies of the versions used for the
native baseline timings.

The measured timing evidence comes from the unprofiled run in
`../../native_baseline/REPORT.md`: q24 discovery took 26.211 ms through the
serial postings path and 3.172 ms through full rescan. NCU counters below are
diagnostic replay evidence, not application throughput.

## Collection

This attempt produced no NCU report. The benchmark rejected `--samples 1`
before launching its CUDA work because it requires at least three samples for a
non-oracle run. The scheduler job failed and verified cleanup. The rejected
attempt is preserved; the separate `serial_postings_q24_ranked_v3_ncu3_20261027`
run has the completed reports.

## Findings

Not applicable: this run collected no kernel profile.

## Identity and limits

- Frozen executable SHA-256: `a628a22579a576d51f942f91d3c0f279ec7c371fdec36f0b454a956e7a44ae7f`.
- Exact indexed source SHA-256: `a3aecefb1ed359342bbc6ffb6843bda079f8a28082030f07510a0e78275e3853`.
- Large-ranked fixture manifest SHA-256: `674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32`.
- Scheduler job `d9b87ab9-94d6-4c06-8e69-6b90e931d740`, attempt
  `8a8e1479-03c8-41b5-a1df-61534f588e34`; terminal failed, cleanup verified.
- This is a profile of the serial baseline only; it does not characterize the
  ordered-disjoint fastpath or imply biological validity of the synthetic data.
