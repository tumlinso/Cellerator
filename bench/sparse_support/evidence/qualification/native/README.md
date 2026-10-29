# Native numerical and integration qualification

2026-10-27, V100 SM70. Native correctness and sanitizer runs use immutable
binary SHA-256 `e5f903954e39f9c0d7f3ab7fdd142a07025cf4b5bd0f0e164f46763035f07a85`.
Each retained identity lists its source hashes and build configuration. The
binary remains in the local diagnostic artifact directory; it is not committed.

| Run | Scheduler job | Result |
|---|---|---|
| Numerical CUDA reference/VJP/CE-GEO | `8c35176a-078c-4ca7-88aa-305346d25203` | exit 0 |
| Memcheck | `25662ed1-befd-461c-9ec4-ddf034df99ed` | 0 errors |
| Racecheck | `930a83c6-9cbd-43fa-b740-3ddd14c0b3c5` | 0 hazards/errors/warnings |
| Synccheck | `c7bc7d76-13da-4688-9fb0-618899ca4bac` | 0 errors |

All four final receipts verify cleanup. The intermediate racecheck block is
retained separately; the later terminal result is the successful run.

The Torch v3 snapshot's 22-test run passed 21 tests, including the packed
forward, SIMT surrogate VJP, recurrent VJP, and native owner/checkpoint checks.
Its added cross-stream test incorrectly selected SGD, outside the supported
Adam contract. The corrected test and 26 existing mechanism regressions later
passed together; see the sibling lifecycle evidence. This failed test is kept
as a distinct record and is not counted as a full-suite pass.

Local CPU qualification also passed four CTest targets: numeric reference,
index/support/gating, physical image builder, and cyclic derivative sampling.
The Torch-free host build installed successfully, and a separate CMake
consumer linked `Cellerator::sparse_support` and ran a condition-program check.
The optional CPU Product2 regressions passed 11 tests. These host checks do not
establish GPU timing or application learning.

The production prepared condition-root map has a separate source/binary
identity and CUDA/sanitizer qualification under
`../../../experiments/shared_conditions/evidence/production_direct_map_20261027_01/`.
The native derivative sampler and conservative proposal-cardinality diagnostics
were added after the numerical binary was frozen; their host tests and binding
tests are separate from that CUDA binary's evidence.
