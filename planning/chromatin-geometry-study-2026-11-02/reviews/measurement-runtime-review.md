# Serialized runtime fallback review

Reviewed the serialized fallback on 2026-11-02 against the requested study-serialization and sampled-quiescence contract. This review is limited to the source hashes below. No GPU jobs were launched and no live ledger or enrollment was modified. Study serialization is not absolute host exclusivity; sampling cannot rule out activity between observations.

## Findings

1. **A confirmation can pass without any during-execution sample.** `scripts/run_packet.py:72-89` takes a before sample, then enters its sampling loop only while `process.poll()` reports the child running. If the child has already exited on the first poll, only before and after observations exist, but `clean` can still be true. Even a loop iteration begun while the child is live can finish its telemetry collection after the child exits. A one-core temporary mock reproduced `clean=True` with two samples. Require at least one complete observation demonstrably obtained while the monitored process is live; otherwise mark the packet ineligible.

2. **Unattributed target-GPU utilization can pass as clean.** In `scripts/run_packet.py:47-67`, utilization is checked only for non-target GPUs, and ownership is checked only for processes returned by `nvidia-smi`. A target with nonzero utilization and no listed process, including before or after the monitored process, produces no issue. A temporary mock with target utilization 90% and an empty process list returned `issues=[]`. Fail closed for unattributed target activity and for missing or unreliable process observations; allow target activity during the run only when it is attributable to the monitored process tree. Sampled telemetry remains bounded by its observation interval.

3. **Lane input reuse can retain a stale worker guard.** For a lane with prior attempts, `scripts/campaign.py:107-113` checks the closure on `--reuse-inputs` but does not refresh `lane/scripts/run_packet.py`. The registration snapshots that lane copy at `:171-177`. An existing lane initialized before the guard was added can therefore run an older, unmonitored packet wrapper even while the current controller reserves it as serialized. Refresh the wrapper independently of closure reuse or bind and verify the reviewed wrapper hash at registration and evidence acceptance.

4. **Guard success is not yet joined to qualification evidence.** The worker records `serial_telemetry_clean` in packet metrics and makes it part of packet success (`scripts/run_packet.py:295-317`). The current `scripts/analyze_pairs.py:19-70` is explicitly calculator-only, and `scripts/kda_evaluate.py:76-81` labels the KDA result preliminary. No inspected evidence path joins a settled serialized ledger reservation, matching successful Scheduler receipt and cleanup, worker result, guard telemetry and actual paired samples. A separate receipt-bound analysis is required before a result can be treated as qualified; this observation does not assert that the preliminary output itself claims qualification.

## Other reviewed behavior

`scripts/make_kda_config.py:88-96` binds the fallback request to the triage job, device, qualified source archive, binary and matched workload digest. `scripts/campaign.py:122-149` checks the config, binary, source and frozen contract identities, and `scripts/kda_evaluate.py:23-39` checks the copied fallback against the runtime UUID and case. These source checks do not establish that an actual packet used the intended wrapper or that any observed receipt is valid.

The shared controller uses one locked ledger. `scripts/campaign.py:201-208,211-252` refuses confirmation while any study GPU reservation is unresolved and blocks subsequent study submissions while a serialized confirmation remains reserving, submitted or unknown. Settlement requires terminal state and verified cleanup (`:325-365`), so the exclusion persists through ambiguous or unverified cleanup. This covers jobs routed through this controller and ledger, not unrelated host work.

The existing `test_serial_admission.py` and `test_serial_telemetry.py` fixtures passed 4 tests on one CPU core. Additional one-core temporary mocks reproduced findings 1 and 2. No live Scheduler behavior, GPU execution, actual telemetry, or evidence result was tested.

## Inspected identities

SHA-256 values under `bench/chromatin_geometry/`:

| File | SHA-256 |
| --- | --- |
| `scripts/make_kda_config.py` | `56d7cadc3af7c699654ee3b62aa8481051117cc1c91f06a3214852d21e6ad8a7` |
| `scripts/campaign.py` | `389eb34c1fde6ebbca48d25e79888ccf7690f9492e4a041a073364bc95993842` |
| `scripts/run_packet.py` | `845449f4d8d3324639490478d19dc06eac8449a1dfd4d06dea682c4557d0ba23` |
| `scripts/kda_evaluate.py` | `caf7fa77064ebe35b67e0db886104f14c57e60b9f71f10ff2c5f9e506bddc48b` |
| `scripts/kda_driver.py` | `5c78d1d462501284d982e9c09cfea03d4b39f8d2dc7f25d0a94e8f6fd0fa0712` |
| `scripts/interference.py` | `291c82869b56d7ab1cfcde56ffff9617f5e41f9c76705ff38ecbc81062131cc7` |
| `scripts/analyze_pairs.py` | `324167b6b476ffbd636ea5c863c80327da966753a6e2d9249b0e3f59dc586344` |
| `tests/test_serial_admission.py` | `5decb53a5051c049853ee653dc451ef833630cdf1a08b4fa3ff3459878e68ee7` |
| `tests/test_serial_telemetry.py` | `e025695412f35a748a90bfc8de211a1b35f8da2c0cf85892d0cd590304da4940` |
