# Combined FP64 integration

This check rebuilds the current canonical checkout's binding consumers and
native arithmetic/moments probes, installs the Python and Torch extensions into
an isolated CMake prefix, then runs the rebuilt consumers under the CUDA
controller. It also invokes the existing FP64 qualification runner. The
installed prefix is the integration input; this check does not build or validate
a wheel.

Run the complete check from the repository root:

```sh
python3 -B planning/fp64/integration_check.py --phase all
```

`--phase build` performs only the CPU-side configure, build, and install steps.
Those steps use four CPU threads, CUDA 12.9, and `sm_70`, with build outputs
under `/tmp/cellerator-fp64-integration-*`. The complete check then asks the
controller to run four C++ binding fixtures, the arithmetic and neighborhood
moments binaries, and exactly 50 Python binding tests with zero skips. The
separate `tests/fp64/run_checks.py --phase all` invocation owns its controller
leases; no controller invocation is nested inside another.

The combined receipt is
`planning/fp64/integration/evidence/combined/integration.json`; per-command
logs and the combined controller receipt are beside it. The controller-held
runtime receipt and logs are under `.../combined/runtime/`. The existing FP64
runner writes its independent receipt and evidence under
`planning/fp64/integration/evidence/fp64/`. The combined receipt records the
source and binary SHA-256 values, commands, return codes, and references to the
FP64 phase receipts. Any failed command, missing binary/module, skipped Python
binding test, or non-passing runner receipt fails the integration.

The runtime requires the configured foreground CUDA controller and the UUID
`GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a`. CPU-only preparation does not
reserve a GPU or run executable tests. Results qualify this source checkout's
integration fixtures only; they do not establish wheel behavior or production
readiness.
