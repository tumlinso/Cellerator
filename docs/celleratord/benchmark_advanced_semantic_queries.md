# Advanced semantic query benchmark

The I02 acceptance benchmark exercises profile propagation, candidate explanation,
Semantic IR rendering, realization decomposition, and source-to-native navigation in
both cold and cached states. Each record reports 31 steady-clock samples, p50 and p95
latency, process peak RSS, and a 10 ms background-work budget.

Cancellation is checked at every query boundary, and the benchmark performs a syntax
completion probe after every series. The focused gate requires that probe to complete
within 100 ms so advanced semantic work cannot make ordinary C++ editing unresponsive.
The measurements are process-local acceptance evidence, not a hardware throughput claim.
