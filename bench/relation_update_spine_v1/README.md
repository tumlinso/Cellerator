# Relation update lifetime measurements

This harness measures the existing prepared relation and its actual providers.
Architecture and arithmetic authority remain in
[the execution model](../../docs/biological_execution_model.qmd) and
[the core execution documentation](../../docs/core_execution_cp_math.qmd).
It produces JSONL samples for the I06 performance review; correctness smoke
results do not promote a route or establish an end-to-end training benefit.

Build `lifetime_bench.cu` as a normal CUDA17/sm70 translation unit and link the
same relation/calculus/readiness/provider libraries as the RU1 GPU tests. It
includes the independent verifier reference header, not any implementation TU.
The integrated target is supplied by I03. A standalone build must separately
compile `src/compute/operation/prepared_relation.cu`; do not include it twice.

Run only under the CUDA controller and repository benchmark mutex. For example,
place this argv in the controller spec (replace the binary path):

```sh
build/ceRU1LifetimeBench --fixture mixed --profile half --route hybrid \
  --modules 16 --horizon 128 --warmup 2 --repeats 7
```

Options:

| Option | Values and meaning |
|---|---|
| `--fixture` | `dense`, `mixed`, `irregular` |
| `--profile` | `half` (explicit f16 RNE operands, f32 accumulation), `full_f32` |
| `--route` | `sparse`, `hybrid` (forced; ineligibility is a nonzero exit) |
| `--modules` | 1–256; default16; scales deterministic 16x16 modules |
| `--horizon` | 1–1024; report matrix uses1,16,128,1024 |
| `--warmup`, `--repeats` | Discarded lifetimes and measured independent samples; default1,5 |
| `--operand-reuse` | `refresh` default: new operand versions each step; `fixed`: versions reusable within a lifetime |
| `--width` | 16 default;1 is the retained forward-only, read-only baseline |
| `--diagnostic` | Additional CUDA events group dense, gradient and update phases; perturbs execution |
| `--profile-one` | Requires `--repeats 1`; profiler API capture around the measured resident lifetime |

Each module contributes a complete 16x16 support block. Dense has isolated tail
axes; mixed adds six irregular tail edges; irregular has two edges per selected
row, including empty rows, and no eligible rectangular cover. Sources are
`16*modules+3`, destinations `16*modules+5`. Edge order is descending within each
dense row, so physical/logical order differs. Initial weights are
`half_RNE((edge_index%31-15)/64)`; input and cotangent use fixed modular formulas
with denominators37 and41. `fixture_fnv1a64` covers little-endian topology words
and exact f32 operand bits; the seed string names the formula version. Record
the source commit and binary SHA256 as well: FNV is an identity, not authentication.

Compare full_f32 sparse with half sparse to expose arithmetic policy, and compare
half sparse with half hybrid to assess the execution route under identical
rounded operands. Full_f32 hybrid and irregular hybrid must fail honestly; they
are explicit eligibility controls, not skipped successful runs. Run all three
fixtures, every declared horizon, and repeated samples. Report medians and spread
from ordinary unprofiled samples, including losses. The default operand-version
refresh charges packing every step even though deterministic operand bytes remain
constant; fixed-version runs separately measure legal packing reuse.

Each N16 step executes forward, transpose, scalar edge gradient, and gradient
step with alpha1/1024 on the same prepared relation. Update and readiness event
publication share the real entrypoint. Topology and gradient preparation occur
once. Before every warmup/measured sample, `publish_values` restores the same
initial logical f16 values into the existing physical plane and waits outside the
timed interval; generation numbers remain monotonic. No host weight master or
structure reconstruction enters the measured lifetime. Input transfer is cold.

Timing fields distinguish:

- `topology_prepare_ms` and `gradient_prepare_ms`: host elapsed cold preparation,
  including each completion wait; cold values repeat in the per-sample rows and
  must not be treated as independently repeated preparation measurements.
- `h2d_ms`: cold input/cotangent/initial-weight transfer and its stream completion.
- `reset_ms`: initial-weight replacement publication and completion outside timing.
- `resident_gpu_ms`: CUDA event interval covering all resident steps, including
  stream idle gaps caused by host submission; this is not a sum of kernel times.
- `resident_wall_ms`: host submission through final event observation, including
  its wait. Both resident times exclude reset, validation and transfer.
- `observation_d2d_d2h_ms`: post-lifetime const lease, D2D copy, reader return and
  D2H observation. N1 has no mutable observation and this field is inapplicable.

`--diagnostic` adds dense-forward+transpose, gradient-including-pack/WMMA/
extraction/residual, and update-including-publication event intervals. Publication
is an event API operation and is not falsely reported as a separate GPU kernel.
Unmeasured subphases are omitted, not emitted as zero. Every sample reports actual
forward/transpose/gradient/update/readiness and sparse/WMMA/residual/pack-refresh
counters plus persistent/scratch bytes. Independent references validate every
gradient and the final physical f16 update recurrence outside timing; N1 output
uses the independent forward loop. A failed numerical or attribution assertion
terminates nonzero. Sanitizer timings are correctness evidence only.

For diagnostic kernel attribution, run Nsight Systems through the same controller:

```sh
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /tmp/ru1-mixed-hybrid \
  build/ceRU1LifetimeBench --fixture mixed --profile half --route hybrid \
  --modules 16 --horizon 128 --warmup 2 --repeats 1 --profile-one
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv /tmp/ru1-mixed-hybrid.nsys-rep
```

The profiler range excludes cold prepare/reset and post-run D2H validation. Use
actual pack kernels, rectangular WMMA, `extract_kernel`, sparse residual and
value-update kernels to attribute inclusive gradient time; use the CUDA API
report for event publication and host waits. Keep profiler times separate from
normal measurements. Record GPU UUID/name/topology, compiler/CUDA/driver/Nsight
versions, binary/source hashes and flags in the external receipt. The expected
limiter depends on fixture and size: launch/host submission overhead for small
supports; packing and dense-operand traffic versus reuse for larger supports.
No performance claim is established merely by these expectations.

## Receipt validation

`tests/relation_update_spine_v1/evidence_contract_test.py` is both a standalone
stdlib tamper suite and an importable bounded receipt checker:

```sh
python3 tests/relation_update_spine_v1/evidence_contract_test.py
python3 tests/relation_update_spine_v1/evidence_contract_test.py \
  --receipt /tmp/receipt.json --expect /tmp/reviewed-expectations.json
python3 tests/relation_update_spine_v1/evidence_contract_test.py --samples /tmp/lifetimes.jsonl
```

The independently reviewed expectation manifest supplies `kind:device_acceptance`,
`source_commit`, `source_dirty_paths` (path-to-SHA256 map), `gpu_uuid`,
`build_config`, and `required_runs`. Each required run specifies `test`, `tool`
(memcheck/racecheck/synccheck), `binary_sha256`, `fixture_identity`, and
`numerical_policy`, and exact `argv`. For I05/I08 the expected inventory must come from the complete
acceptance matrix: memcheck all GPU/demo binaries, and racecheck/synccheck for
numerics, lifecycle and WMMA legality. A bounded local receipt is not that final
inventory. Expectations must not be copied automatically from an unreviewed
receipt to make it pass.

The receipt repeats the expected source/build fields and records
`gpu_compute_capability:"7.0"`, observed `gpu_name`, `gpu_uuid`,
`device_executed:true`, `skipped:false`, `tool_versions` with compiler/CUDA/driver/
compute_sanitizer, and `build_argv`. `controller_lease` and `controller_log` each
contain `path` and `sha256`; these name the actual CUDA-FOREGROUND-LEASE/1 receipt
and raw controller stderr proving benchmark-mutex acquisition/release. Artifact
paths resolve relative to the receipt directory; absolute paths are allowed.
Dirty source contents and binary bytes are rehashed against external expectations.

Each entry in `runs` records its expected test/tool/fixture/policy fields,
`status:"completed"`, `executed:true`, `skipped:false`, integer `exit_code:0`,
`binary:{path,sha256}`, exact `argv`, and raw `log`, `stdout`, `stderr` strings
with corresponding `log_sha256`, `stdout_sha256`, `stderr_sha256`. The argv must
name the actual tested binary and sanitizer tool. The checker requires complete
zero-error summaries, rejects any conflicting nonzero summaries or race warnings,
and checks exact nonduplicate execution coverage. Raw files may be embedded by a
receipt producer; the existing sanitizer wrapper already captures these strings.
Missing GPU, missing/skipped sanitizer, timeout, source-only records, incomplete
logs and a bare `passed:true` cannot establish a passed device acceptance.

This detects incomplete evidence and accidental tampering against separately
reviewed expectations. It is not cryptographic remote attestation against a
malicious author who can forge all artifacts and the independent expectations.
