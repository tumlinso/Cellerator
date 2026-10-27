# Tokenization qualification and benchmarks

The public API is documented in [ATen tokenization](../../docs/tokenization.qmd).
Tests use synthetic inputs and independent numerical expectations. They do not
infer biological events or establish biological validity.

## CPU qualification

Use the interpreter where the tokenization-enabled Cellerator wheel is installed:

```sh
python -m unittest discover -s tests/tokenization -p test_tokenization.py -v
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python bench/tokenization/benchmark.py \
  --device cpu --rows 32 --features 20000 --repeats 5 --append-calls 32
```

The CMake test names are `cellerator_tokenization_numerics`,
`cellerator_tokenization_core` and `cellerator_tokenization_append`. Enable
`CELLERATOR_BUILD_TESTS` when configuring that build before running CTest.

## CUDA qualification and profiling

Run all CUDA commands inside finite Device Scheduler jobs with a UUID-selected
exclusive GPU grant, declared memory estimate and `gpu.quiescence` precondition.
Use Cellerator's separately scoped scheduler client configuration. GPU visibility
and an installed profiler do not establish execution admission.

The admitted job's environment enables GPU tests explicitly:

```sh
CELLERATOR_TOKENIZATION_TEST_CUDA=1 python -m unittest discover \
  -s tests/tokenization -p test_tokenization.py -v
compute-sanitizer --tool memcheck --error-exitcode 99 \
  python bench/tokenization/profile_smoke.py --steps 4 --sequences 8
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output=tokenization-trace python bench/tokenization/profile_smoke.py
ncu --kernel-name 'regex:.*append_copy_bytes.*' --launch-count 1 --set basic \
  --export=tokenization-kernel python bench/tokenization/profile_smoke.py
```

Use HPC SDK CUDA 12.9 tools explicitly on the qualified host; CUDA compilation
uses SM70 and `-lineinfo`. Profiler replay results are not throughput timings.

## Timing scope

The driver checks output equality against NumPy and equivalent ATen dense and
sparse compositions before timing. Sparse baselines operate on stored entries.
It records input transfer/projection, numerical fitting, rank/pack, sequence
buffer setup, and steady-state append separately. The append comparison uses
preallocated ATen destination/scratch and a reusable Cellerator status tensor.
It does not rebuild history on each call. Device synchronization bounds each
timing series; it is not inserted into every append call.

Keep per-workload results, compiler flags, source hashes, scheduler receipts and
cleanup evidence together. These fixtures include both validation overhead and
algorithm differences; a result on one density or batch size does not establish
a universal performance advantage.

The dated qualification summary under `evidence/` records the delivered source
and actual results. Full build logs and profiler reports live under the ignored
`build/tokenization-evidence/` directory.
