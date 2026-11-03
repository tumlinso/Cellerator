# Measurement review follow-up

Reviewed 2026-11-02 against the three findings in `measurement-review.md`. This is a review of the versions identified below, not a review of the live interference results. No GPU job, live enrollment, or ledger was modified.

## Remaining findings

1. **Concurrent subprocess overlap does not establish concurrent measured work.** `scripts/interference.py:311-325` timestamps the entire runner subprocess; `:749-756` accepts any positive overlap of those windows and the Scheduler attempt windows. The runner performs preparation, full-chain validation, warmups, and measurement inside that subprocess (`campaign_runner.cu:739-762,971-997`), so even a tiny overlap confined to validation or non-corresponding phases can pass while the measured operational phases run separately. A barrier before subprocess launch does not prove the required overlap. Record the relevant execution-preparation and timed horizon/recovery windows, and require adequate overlap for the comparison being claimed, or leave interference inconclusive and use a separately enforced serialized measurement policy.

2. **The `serialized` status does not establish or enforce host-wide serialization for confirmation.** `scripts/interference.py:592-625` accepts a descriptive policy and at least two non-overlapping jobs on the same primary UUID. Device-exclusive scheduling already serializes those jobs with each other; this check does not prove the absence of work on other GPUs or enforce the stated whole-job policy for subsequent confirmation packets. At `:803-808`, the serialized branch can also be selected when isolated telemetry is not clean. `scripts/make_kda_config.py:138-144` and `scripts/kda_evaluate.py:31-37` accept that status. Require actual enforceable host-wide serialization and evidence covering confirmation jobs, or keep the measurement ineligible for confirmation.

## Resolution of original findings

* **Failed packet export:** resolved for the reported case. `scripts/export_correctness.py:23-63` joins settled packet and receipt identity, classifies non-passes into `failures`, and continues to later successful packets. Lines 109-124 additionally classify settled failed packets without metrics. The new isolated fixture test exercises one failed and one passed packet together. This does not attest to any particular live packet.
* **Interference evidence identity and joins:** materially improved, but not fully resolved because of the timing and serialized-policy findings above. The analyzer checks retained closure and binary hashes and a matching build receipt (`scripts/interference.py:628-649`), frozen case/input/operation identity and raw campaign cost fields (`:657-698`), unique successful cleanup-verified Scheduler job/attempt/UUID receipts (`:476-509,697-718`), and distinct paired barrier groups (`:722-767`). It checks isolated-job nonoverlap and sampled all-device activity (`:520-546,768-787`); sampling cannot exclude short unobserved activity. `make_kda_config.py:88-144` requires unique job IDs and reconstructs the analysis from retained raw packets using the analyzer inside the qualified archive. Before any conclusion, the actual jobs still need successful settled receipts, retained raw packet and binary/closure evidence, adequate phase overlap, clean isolated observations, and a supported statistical result. A status flag or these mock tests is not that evidence.
* **Complete host/replay cost:** the previously missing components are now explicitly measured for the stated fixture boundary. `campaign_runner.cu:739-753,762-768,1085-1097` times re-preparation of the consumed host address layouts and includes condition preparation in cold device preparation, with layout verification outside the timed interval. `data/replay.cc:31-102` includes replay hashing, reading, decoding, and applying changed input observations, and separates reference recomputation. `kda_driver.py:125-140` requires the replay + host layout + device preparation sum and adds recurrent horizon and output recovery. The logical workload and test payload construction is still treated as supplied deterministic fixture input, and independent reference work is excluded; a different data-ingestion boundary would need its own accounting. This is a source-level boundary assessment, not a timing or performance validation.

The one-core CPU tests `test_interference.py` (13), `test_export_correctness.py` (1), and `test_kda_driver.py` (1) passed using temporary mocked fixtures. They do not establish actual Scheduler behavior, interference absence, or GPU correctness.

## Inspected source identities

SHA-256 values, under `bench/chromatin_geometry/` unless otherwise stated:

| File | SHA-256 |
| --- | --- |
| `scripts/campaign.py` | `d8b15598742e39da89a20768451b9de9f87088244222f3861b91c3dc353c5015` |
| `scripts/export_correctness.py` | `b565977a74244ceec055b5000728538e7d92b533da1367db158eecd37d868d53` |
| `scripts/interference.py` | `9c7b17424925a33432a707b671aea910f3fb4a543be98749decc85290a9fbc49` |
| `scripts/make_interference_config.py` | `41c78b8712d78dde6a1d0b0f5ec035cae024f7c23a039673470c3ea5406e5efd` |
| `scripts/make_kda_config.py` | `5f9f2e763159bc1612ac094f02d94443da915a6f6593871a19d519e3b7f04814` |
| `scripts/run_packet.py` | `006e7287d047495cc333e1682d7e95895c86c6c4565e65cece638ac432f5f116` |
| `scripts/kda_driver.py` | `5c78d1d462501284d982e9c09cfea03d4b39f8d2dc7f25d0a94e8f6fd0fa0712` |
| `scripts/kda_evaluate.py` | `b7db19dcf9e40055ff98b633c25c3c4ad32a44f5884a57715ed73cd3ca6fe731` |
| `campaign_runner.cu` | `aa3f6c3cde61a3cc1be64b707a82d0ec983606fc92e5c84d37d6dcf8614d7624` |
| `workloads.cc` | `79aedb356e240549d1ced02c92dfc5d493c9529aa30cdcf3e2b3a9bf524e3c24` |
| `data/replay.cc` | `029dfe27959b8d1fbba072394675e2b369a13f624aad9715770e0b8ed7dc4279` |
| `backends/conditions.cu` | `f0299db6a02599bffe77005aa01326a71035f1e6e11d18827bb5a1388972c393` |
| `tests/test_interference.py` | `72e237aef58368f275b845333f7c42e901160c9541d726ffbb6e4a404ef91964` |
| `tests/test_export_correctness.py` | `b4e99f4b91b12469249b650dbff36c2a8c648167f35966fc8a4158c5f6a26e0e` |
| `tests/test_kda_driver.py` | `99c04bd33ee04295c73169a1a9dfe753e359eba5441d68dec12ed1c60f96fecc` |
| `planning/chromatin-geometry-study-2026-11-02/contract-freeze-v2.json` | `237da579f5068fc26469f52c79b7ae33beb1ab0af057129a27c186193c5c8147` |
