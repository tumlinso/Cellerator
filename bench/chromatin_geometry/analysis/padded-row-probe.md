# Held probe: padded shared feature rows

This is an unimplemented, profile-gated hypothesis. The frozen unpadded
candidate remains the baseline. No benchmark or candidate is registered by
this note.

In `Values::argument`, lanes of a cohort warp own cells and read the same local
feature slot from different shared rows. The address in FP32 words is
`buffer_base + cell_position * row_stride + feature_slot`. NVIDIA documents
32 shared-memory banks with successive 32-bit words assigned to successive
banks, and distinguishes distinct-word conflicts from same-word broadcasts.
[CUDA 12.9 Best Practices, shared-memory banks](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)

The following is an inference from that contract and this kernel's address
map, not a measured conflict count: stride64 sends active lanes to distinct
words of one bank for a fixed feature. Stride49 has step17 modulo32 and visits
distinct banks for consecutive lanes. Small cohorts or inactive groups reduce
the number of participating accesses. Uniform parameter reads are same-word
broadcasts and should not be confused with the feature-row issue.

If the V100 profile shows meaningful shared-load conflicts/wavefront overhead
on the complete path, test only this physical change first:

```text
row_stride = max_features == 0 ? 0 : (max_features | 1)
```

Thus64 becomes65 and49 is unchanged. Apply the stride consistently to staging
stores, `Values` reads, buffer offsets, the parameter-stage base and the
capacity calculation. Keep feature counts, membership masks, local argument
indices, cell/cohort order, global X addresses, gate contracts, arithmetic,
rounding and synchronization unchanged. Never load padding. The extra storage
is `tile * buffer_count * sizeof(float)` for the64-feature cases; tile128/two
buffers remains rejected. Retain the unpadded result for attribution.

A study-only build switch is sufficient; it can preserve the frozen call ABI.
Record the switch, source, binary and input identities in each candidate packet.
Do not fold unrelated cohort scheduling, weight staging or arithmetic changes
into this test.

Acceptance remains the independent scalar/bound checks, exact descriptor
membership/identities, poison/tail/zero-arity cases and all sanitizer modes.
Profile both variants with the same cells, gates, tile, buffers and device.
Check actual shared conflicts and wavefronts, global/L2 traffic, spills,
resource use and complete latency. A reduced conflict counter alone does not
establish a useful specialization: retain the study's >5% improvement and
paired95% interval rule against the strongest matched control. Stop if another
phase dominates or the GPU budget cannot support confirmation.
