# libCellerator and celleratord benchmark v1

Ordinary-C++ and Cellerator documents separately measure API session startup,
concurrent parsing, cancellation, editor startup, diagnostics, completion,
hover, IR/candidate queries, and peak memory. Report median and p95 from eleven
raw samples under the benchmark mutex. Ordinary-C++ latency is an explicit
regression guard; Cellerator work cannot be charged to unactivated files.
