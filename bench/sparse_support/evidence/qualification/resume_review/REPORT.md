# Independent restart review: postings and CE-GEO validator

## Scope and source identity

Reviewed the production delta from `4452dc117ad4b5240602d754506431f5f859d5f0`
through restart commit `8fa6c09ec72fdc04e474ff1e022324668cceeb51`:

- `src/compute/indexing/cuda_postings.cu` SHA-256
  `3a6df1882e7feb318aa44b2fc4ba66c42a619f8afa7b452612c1a84fd0f14381`.
- `include/Cellerator/compute/indexing/cuda_postings.cuh` SHA-256
  `55fb929f0ce96427253bc2a4132624b503d1f13ce315ee2642dc17cbd0db1bc2`.
- `src/compute/projection/mma_physical_cover_validation.cc` SHA-256
  `03532ce1bfc632312ae0802381c6cd06ec6c01f91718cc41238b6c94d5328254`.
- `tests/sparse_support/cuda_worksets_test.cu` SHA-256
  `6f8f37668c242465cec88c5d5433822d8a2a3f21a0c5424169d997f51e688ed8`.

The source hashes match the pause checkpoint. No production source was changed.

## Findings

### Ordered-disjoint postings

No correctness blocker found for canonical projected posting inputs. The host
projection contract supplies each posting slice in stable edge-ID order. The
small-batch path admits concatenation only when query terms are strictly
increasing and unique and every nonempty slice begins above the previous
nonempty slice's last edge ID. Under those conditions concatenation is already
the sorted set union. Duplicate, unsorted, or overlapping rows use the existing
k-way merge. Missing terms are empty slices; the small path safely skips them,
and the parallel multi-term path conservatively falls back when any slice is
empty.

For the parallel path, one CTA scans at most `128 * 16 = 2048` terms. It checks
strict key order and disjoint adjacent posting ranges across all terms, then
uses CUB's block exclusive scan for each thread's aggregate and adds the local
prefixes. Rows above 2048 fail eligibility before the bounded scan and use the
exact merge over the full row. Query-tail loops stop at `qi >= qe`; the
one-term and empty-row cases have separate branches. The scatter copies each
slice only within the row's counted range.

The independent exhaustive host property check passed **163,840 query rows**
across **4,096 assignments** of three sorted posting lists. It compared eligible
concatenation with sorted set union and checked fallback cases against the
union oracle. Prefix arithmetic passed lengths 0, 1, 15, 16, 17, 127, 128, 129,
2047, and 2048; 2049 selected fallback. The script is
[`ordered_disjoint_proof.py`](ordered_disjoint_proof.py). This validates the
precondition's mathematics and scan-prefix arithmetic, not the CUDA execution.

Representable capacity overflow is checked in `prepare_offsets_kernel` before
it writes the offset array; the subsequent scatter kernel returns while status
is nonzero.
The existing focused GPU test checks unchanged item/offset sentinels for an
undersized output, and the retained q2048 scheduler result records exact
required count with no partial write. Separately from this independent
CPU/source review, the resumed GPU3 focused test and Compute Sanitizer
memcheck, racecheck, and synccheck all passed with cleanup verified under
commit `18c87132`; scheduler receipts/logs and exact binary/source hashes are
retained in the ordered-postings checkpoint. I did not launch those GPU jobs.

The cursor scratch record changed from two 64-bit fields to a 32-byte public
record. In-repository callers allocate by `sizeof(cursor_v1)` or a typed buffer;
no hard-coded old byte size was found. This is source-compatible after rebuild,
but not binary-compatible with an already compiled caller using the former
layout. The header and pause note tell callers not to rely on that layout. If
the `v1` type is intended as a frozen external ABI, versioning is a release
follow-up; no such external ABI promise was found in this review.

The original source snapshot had an unchecked aggregate arithmetic boundary:
`status.required` used 64-bit `atomicAdd`, and `prepare_offsets_kernel` added
row counts without an overflow test. Multiple valid cells can query the same
posting list, so per-row bounds do not bound the aggregate. The follow-up patch
now rejects nonempty shapes when `cell_count * index.item_count` is not
representable as `uint64_t`. Since any per-cell union contains at most all
indexed items, this conservative prelaunch bound protects both the device-side
required total and offset prefix sum without adding hot-kernel arithmetic. It
may reject some sparse query shapes whose actual output would fit, but makes
the accepted shape bound explicit.

### CE-GEO physical-cover validator

The asymptotic rewrite replaces repeated logical-ID scans with a byte owner
array and residual overlap pair scans with sorted interval checks. The added
physical-slot owner marks each map slot once, checks every occupied slot's
back-reference, and rejects duplicate `(tile, dense_slot)` keys. These checks
preserve exact image bounds and require no serialized ABI change. Vector
allocation/sort work is inside the `noexcept` catch boundary, so allocation
failure rejects instead of terminating.

The retained differential probe passed on the reviewed sources: 96 generated
valid images; 288 malformed images rejected identically by old and new
validators; 96 ignored alignment-padding cases accepted by both; and 96
duplicate physical-slot images accepted by the frozen old validator but
rejected by the new one. I added and compiled a small adversarial C++ probe
that independently constructs a duplicate-slot image; it confirms the same
old-accept/new-reject result. See
[`validator_adversarial_probe.cc`](validator_adversarial_probe.cc). The source
reference is the frozen validator at commit `4452dc11`.

## Evidence and limits

- `python ordered_disjoint_proof.py`: pass.
- `g++ -std=c++17 -O2` standalone build and execution of
  `validator_adversarial_probe.cc` linked against the frozen validator,
  current validator, and image builder: pass.
- Retained `validator_differential_probe`: pass with the counts above.
- No GPU job or full test suite was launched by this review. The separate
  completed GPU3 sanitizer evidence is summarized above and in
  `docs/sparse_support_foundation/ordered_postings_fastpath_resume.md`.

The independent checks add evidence for precondition sufficiency, CTA scan
boundaries, and physical-slot rejection. The separate sanitizer run establishes
memory, race, and synchronization safety only for its focused GPU3 test paths,
not every possible input size or long-running workload.

## Aggregate-bound follow-up

After this review, the owner added a prelaunch representability guard and a
`--bound-preflight-only` test branch. I independently inspected the narrow
change: division-based checking avoids overflow, and the guard precedes all
pointer checks and kernel launches. The test uses `cell_count=2` and
`index.item_count=UINT64_MAX`, expects `cudaErrorInvalidValue`, then exits
before any CUDA allocation/runtime call. The owner reports a successful
incremental SM70 build and host-only branch execution; evidence is retained in
`bench/sparse_support/evidence/experiments/ordered_postings_fastpath_checkpoint/overflow_bound_host_preflight.txt`.

The follow-up source identity is `cuda_postings.cu`
`d034ac6815e070b356d08b3f8852cf03d7ab39a91d008ab12008bbf30c92f99c`, public
header `6ebcc3a4f91d57ece2b66f9dc4f0db23d998d5f6d78a84d2bb240547adc73a80`,
and CUDA test source
`14b1cad0875387d46984cdd7aa37eb6abec529d23af5905f832959adcc1510a5`. The
rebuilt test binary SHA-256 is
`7e1dbadc31f5af4c582618d6aa04848db2b06c3292d6b6f8304d4be8c7f4e122`. This
focused host preflight check does not rerun the GPU tests or sanitizers; those
remain evidence for the prior kernel source identity, with the guard itself
separately verified to return before GPU work.
