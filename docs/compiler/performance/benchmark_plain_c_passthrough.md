# Plain C++ passthrough benchmark v1

The harness crosses small, medium, and template-heavy corpora with direct GCC
and direct Clang baselines. Each cell captures driver overhead and preprocess,
compile, and link wall time, peak RSS, depfile bytes/content, object size,
diagnostic count/severity, and exit status for eleven raw repetitions after two
warmups. Inputs and flags are byte-identical. Runs acquire the benchmark mutex;
missing compilers are reported as unavailable rather than substituted.
