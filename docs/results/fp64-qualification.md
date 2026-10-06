# FP64 prepared relation qualification

## Scope

The qualification target compares prepared forward and transpose application
against SciPy CSR for all supported FP32/FP64 relation-feature tuples and
feature widths `1, 3, 15, 16, 17, 33, 65, 257`. Fixtures contain signed values,
duplicate edges, irregular rows, an empty row, large cancellation terms, and
two successive value generations. The native target also covers FP32/FP64
elementwise multiplication, in-place squaring, affine combination, and
partial-overlap rejection. See [the runner](../../tests/fp64/README.md) for
build and controller commands.

## Evidence

| Check | Status | Evidence |
| --- | --- | --- |
| CUDA build | passed | [gate record](../../planning/fp64/evidence/run_checks.json) |
| NumPy/SciPy oracle | passed, 194 cases | [oracle record](../../planning/fp64/evidence/scipy.json) |
| Native CUDA witnesses | passed, 6 targets | [native record](../../planning/fp64/evidence/native/native.json) |
| Compute Sanitizer memcheck | passed, 0 errors for qualification and admission targets | [qualification receipt](../../planning/fp64/evidence/controller/sanitize_qualification.json), [admission receipt](../../planning/fp64/evidence/controller/sanitize_admission.json) |
| Preparation, transfer, resident execution, end-to-end timing, and memory | measured | [raw cost record](../../planning/fp64/evidence/costs.json) |

The oracle covers 128 prepared-relation cases, 64 shared-CSR duplicate-edge
cases, and 2 variance cases. The prepared cases use all four supported tuples,
both orientations, two value generations, signed values, irregular and empty
rows, and feature widths `1, 3, 15, 16, 17, 33, 65, 257`. The prepared masked
topology still rejects duplicate endpoints within a row; duplicate-edge
coverage applies to the shared CSR traversal.

The variance identity had zero centered-reference error in the well-conditioned
case, with a `4.99e-13` roundoff bound. In the cancellation-heavy case, its
error was `0.5` against a conservative `115.46` reduction-order bound. The
second-moment subtraction therefore lost the variance even in FP64; the bound
does not guarantee stability. The SciPy comparator uses `3e-13` absolute and
relative tolerances plus a contribution-mass reduction bound for cancellation
cases; the FP32-only tuple uses `2e-6` absolute and relative tolerances.

Costs were measured on a Tesla V100-SXM2-16GB with CUDA 12.9, at width 65 for
2048×2048 axes and 32,768 edges. Each timed phase used 20 warmups and 200
repetitions. Times below are microseconds except preparation and end-to-end;
device-persistent bytes exclude the estimated 655,360-byte host topology.

| Relation × feature → output | Forward / transpose | H2D + publish / D2H | Prepare / end-to-end | Relation values / requested device-persistent bytes |
| --- | ---: | ---: | ---: | ---: |
| FP32 × FP32 → FP32 | 9.754 / 9.784 µs | 325.427 / 80.781 µs | 15.938 / 17.248 ms | 131,072 / 1,802,056 |
| FP32 × FP64 → FP64 | 10.399 / 10.194 µs | 606.735 / 160.043 µs | 16.272 / 16.167 ms | 131,072 / 1,802,056 |
| FP64 × FP32 → FP64 | 9.856 / 9.866 µs | 533.468 / 167.108 µs | 15.595 / 15.986 ms | 262,144 / 1,933,128 |
| FP64 × FP64 → FP64 | 10.368 / 10.004 µs | 624.579 / 162.258 µs | 16.115 / 16.640 ms | 262,144 / 1,933,128 |

The provider reported zero temporary device bytes for resident apply and
elementwise execution. The direct FP32 comparison matched the historical
kernel output exactly and measured a `-0.269%` change, below the `5%`
investigation threshold. This single workload is not a broad performance claim;
end-to-end preparation timings also show order effects, so no FP64 speedup is
claimed. FP64 value storage doubles for FP64 relation weights, while promotion
to FP64 features/output increases caller-buffer and transfer bytes; see the raw
record for staging and complete byte counts.
