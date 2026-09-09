# T02 independent numerical stress fixtures

`tests/native_foundation/reference/t02.cc` executes 1137 checks and 16 deliberately
wrong numerical variants. The reusable scalar helpers in `stress_fixtures.hh`
contain no production headers, layouts, or provider lowering. They complement the
T01 nonlinear FP64 formulas and remain reference fixtures, not production runtime
or CUDA qualification.

| Case | Expected behavior |
| --- | --- |
| Widths 0, 1, 16, 31, 32, 33, 65; segment degrees 0, 1, 4097 | Exact dyadic FP64/FP32 sums, empty sum zero, untouched outer canaries; width32 truncation rejected |
| Signed additive destinations | Exact sum; last-writer overwrite rejected |
| Sequential FP32 cancellation vs FP64 | Explicitly different exact results 0 and 1; profile substitution rejected |
| FP16 storage then FP32 arithmetic | Round nearest, ties to even; normal/subnormal limits, signed underflow zero, finite maximum and overflow classification |
| Storage error budget | Sum of independent half-ULP storage bounds, kept separate from arithmetic error; dyadic quantized fixture sums exactly in FP32 |
| FP32 overflow | Positive infinity preserved, distinguished from finite FP64 diagnostic result |
| Exact masks and nonfinite inputs | Exclude inactive entries before arithmetic; active NaN and signed infinity propagate, opposing infinities produce NaN; sanitization rejected |
| Tiny support contribution | Exact 2^-40 term retained; threshold pruning rejected |
| Repeated argument arities 1, 2, 3, 9 | Exact product and every incidence in the derivative; deduplication rejected |
| Near-zero primal and quantization derivatives | Small coefficient sensitivity retained; mathematical derivative at stored values distinct from derivative through rounding |
| Unsupported FP8 profile | Explicit failure, no V100 arithmetic claim |

Finite comparison is explicit absolute plus relative tolerance. Nonfinite
comparison checks NaN classification and infinity sign; a finite sanitized answer
cannot satisfy it. Mask extent and nonbinary-mask failures are exercised.

Prepare the clean committed source with `prepare.py --bindings <external-json>
--target ce_nf1_t01 --target ce_nf1_t02`, then run the authoritative T02 gate.
The preparation receipt validates CMake's actual source directory, committed HEAD,
unchanged external bindings, and final source cleanliness. Two compiler jobs bound
these two small test translation units. Later provider conformance tasks must bind
these cases to real production execution; this milestone does not claim that.
