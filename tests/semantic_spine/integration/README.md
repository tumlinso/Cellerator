# Integrated Semantic Spine conformance

I05 integrates the independent tests from the four producer lanes through
[the root test targets](../CMakeLists.txt). The real cross-origin executable is
built from [integrated_probe.cc](../verification/integrated_probe.cc) and
[contract_probes.cc](../verification/contract_probes.cc); its reference is the
independent [logical-edge oracle](../verification/oracle.hh).

The required command is:

```sh
python3 planning/semantic-spine-v1/scripts/run_gate.py --group conformance --source-root . --build-dir build-ss1 --jobs "$(nproc)"
```

Run it under the repository's canonical CUDA controller lease and benchmark
mutex, with `CELLERATOR_SS1_GPU_LEASE_RECEIPT` pointing to the observed live
lease receipt. It builds and runs the core, algebra, frontend, native and
cross-origin executables, and fails if the device is unavailable. The separate
negative controls are:

```sh
python3 tests/semantic_spine/verification/run_controls.py --build-dir build-ss1
```

They require the wrong-output execution and the missing-device execution to
fail for their expected reasons. The wrong-output control also requires the
live GPU reservation. The source/native witness covers two fixtures, both
orientations and two value generations. See
[origin_execution.json](../../../docs/semantic_spine_v1/origin_execution.json)
for executed commands and evidence, and
[completion.json](../../../docs/semantic_spine_v1/completion.json) for final
acceptance. This directory is the integration entry point; implementations
retain their owning lane paths.
