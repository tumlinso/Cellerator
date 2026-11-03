# Measurement infrastructure review

Reviewed 2026-11-02 against `protocol.md` and `bench/chromatin_geometry/docs/draft.md`. This is a source-bound review, not qualification of an execution or performance claim. The exact inspected identities are below; concurrently edited or later captured versions require reconciliation. No GPU jobs, live ledger mutations, or enrollment changes were made.

## Findings

1. **The interference prerequisite does not establish an isolated-versus-concurrent comparison.** `scripts/make_kda_config.py:84-94` accepts a caller-supplied `passed` or `serialized` status and two receipt IDs per list. Its `receipt()` at lines 20-32 checks ID, cleanup and one matching GPU allocation, but not terminal success, packet/measurement identity, workload, actual concurrent overlap, or distinct jobs. The same unrelated receipt may appear twice in both lists. `scripts/kda_evaluate.py:24-35` checks the attestation and list presence rather than joining measured results. Consequently, the confirmation path can proceed without the protocol's comparison. Join distinct retained measurement packets to successful Scheduler receipts, identify their matched source, binary, input and timing mode, and substantiate the isolated/concurrent comparison or the declared serialization before confirmation.

2. **A retained failed packet prevents the correctness index from being exported.** `scripts/campaign.py:313-326` copies each settled packet into `runs/packets`, regardless of its worker status. `scripts/export_correctness.py:26-52` scans that directory and raises on a failed packet before reaching its separate `runs/failures` scan at lines 82-98. Therefore, after any failed packet is retained through this controller, the exporter cannot generate an index that contains later successful checks; it also does not classify that failure there. Classify retained failures alongside successes using their receipt and raw artifacts, without treating them as passed evidence.

3. **The complete-cost boundary remains unproven for host execution preparation.** `campaign_runner.cu:605-610` performs workload construction, condition structure and host consumer fixture construction before the `host_prepared` timestamp. `workloads.cc:233-290,349-367` shows that this interval combines structural construction and independent oracle work. The reported `cold_execution_preparation_ms` begins after all of it (`campaign_runner.cu:933-937`), and `scripts/kda_driver.py:130-134` uses that field in its full-cost sample. The independent test oracle should be excluded, but the current split does not demonstrate that all excluded construction is test-only. Separate and time operational host preparation, or demonstrate from the declared input boundary that none is required, before claiming complete cost or break-even.

## Acceptance and coverage limits

`analyze_pairs.py` explicitly labels its output calculator-only. It takes caller-provided timing, validation and interference flags and does not join per-arm timings to raw source, input, workload, device, admission, validation, and cleanup records. Its balanced-order and >5% latency gates therefore do not qualify a specialization by themselves. A receipt-bound analysis joining the actual paired KDA outputs remains required, together with evidence that the matched control is the strongest applicable one. Profiles remain diagnostic and are not independent timing samples.

The controller uses a shared locked ledger, reserves before reconciliation, retains unresolved reservations, and checks exclusive GPU allocation and verified cleanup at settlement. Its registration selects one UUID-specific exclusive GPU and a CPU-only build; the inspected Autobench engine specifies one Scheduler attempt and an adapter quiescence precondition. These are source observations, not an audit of live enrollment, all historical receipts, or execution cleanup. KDA preparation is run in the build phase; the inspected local KDA engine's `evaluate_candidates` invokes `prepare_candidates` again, while `kda_driver.py:35-41` fails closed if a cache miss invokes the build adapter in a GPU packet. Thus a missing cache should fail instead of rebuilding, but this review did not execute the admission or cache-hit path and does not certify the actual captured engine and workspace.

The existing CPU-only fixture suite passed 13 tests using one CPU core (`taskset -c 0`, `PYTHONDONTWRITEBYTECODE=1`); these include mocked budget operations, isolated capture fixtures, a temporary synthetic KDA preparation, triage envelope tests, and paired calculator tests. They do not establish the interference comparison, actual Scheduler behavior, raw packet retention, scientific correctness, or timing results. No kernel or scientific-claim review was performed.

## Inspected source identities

SHA-256 at inspection start and end was unchanged for the following study files. Paths are under `bench/chromatin_geometry/` unless otherwise stated.

| File | SHA-256 |
| --- | --- |
| `scripts/kda_setup.py` | `8fb50bbce5cc09793c34606be8ae3c84051e4ab0e43000f43d163fdbf54d5f0d` |
| `scripts/kda_driver.py` | `0dae7d12262969f97e8482763678bb97055f6a8a001fc0b6f825cb7b36cd1b50` |
| `scripts/kda_evaluate.py` | `2c30ac7f550486ea60df868ce02463124532d3ad659e8d8a91fedb05a2266d05` |
| `scripts/run_packet.py` | `da95ec4cb5572f1bf0c741dffee7cf20cb7caaf36d483eda51b5893a971fab48` |
| `scripts/campaign.py` | `2a56f7b9b8aa7aa29a66d7e1233daa492b5afebdcfc66bed2d7379c7d7cf7cfd` |
| `scripts/analyze_pairs.py` | `324167b6b476ffbd636ea5c863c80327da966753a6e2d9249b0e3f59dc586344` |
| `scripts/make_triage_config.py` | `b519131ac5d13b86141f8069649c158c2679e8936488922a965ee804a3755466` |
| `scripts/triage.py` | `cfa9f8ba5e04b251e6ea679f7e88d71df9392bdc0f620e5fc5627989fbd94928` |
| `scripts/export_correctness.py` | `49c60a9c19a7e2f2f9afe2387c2ecd28c04b2ef956f77c3bc1ab532b303a0227` |
| `scripts/make_kda_config.py` | `2e25adead8e021e8b16a4712053c7c3cd7671651aafc35c0a3153fdcf6f27c60` |
| `campaign_runner.cu` | `f50449a8170438d955446dce778e681d7e3880dd342b9b5e38d794536f6053e7` |
| `tests/test_kda_setup.py` | `1b29c22413f7fd694a3f7b26e51bc36ecf5c5fcfc8144692826f894c9218d430` |
| `tests/test_paired_analysis.py` | `5a61ea18c181920a5a3d37e9796b758bd5e7180de239926efaeeb32b0e33a219` |
| `tests/test_repro_controls.py` | `f1baab17e69e0e29a26b9efa4b3cb07be2f66baa5993f89c5f246ea8326ebd62` |
| `tests/test_triage.py` | `6e9a3e295fcffd38d603bfbefe704a323d071cf5a6e0f61c700668146e0007cf` |

The local `/home/tumlinson/kda/kda_experiments/engine.py` observed for the cache-path interpretation had SHA-256 `3c8603f52fec024f1ab5edb5a3b5bfcf2d96ba06e29c89ce91784c70b29c347a`. The protocol and draft hashes at review start were respectively `ab71c0b8a2c3ae7ed79de4a400ca3f66a2ac2632aed3b9e5666400914de1c125` and `f43fa53410ac91a16d4e1224e751c4eea21f830eac2377caa4c9f6515e118108`.
