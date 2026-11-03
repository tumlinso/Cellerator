# Reproducibility and replay verification

Independent review scope: study source capture, Autobench/Scheduler resource
accounting, and the native CGATAC01 replay prepared from the downloaded 10x
PBMC filtered feature matrix. No Project Control authority or live campaign
ledger was modified by the isolated control tests.

## Checks reproduced

`python3 -m unittest bench/chromatin_geometry/tests/test_repro_controls.py -v`
passed 8 tests. They use temporary directories and mocked Autobench responses
to verify that a reservation exists before submission, multiple inactive lanes
share the same limit, an aggregate-budget overrun and unresolved attempt block
submission, an ambiguous submission retains its full reservation, and cleanup
or a mismatched GPU UUID cannot release the reservation. Capture checks reject
symlinks, stale extraction roots, archive-hash mismatches and path traversal.
They do not call the live Scheduler or alter the study enrollment.

I independently compiled and ran
`h5c++ -std=c++17 -O0 -o /tmp/verify-replay-source bench/chromatin_geometry/tests/verify_replay_source.cc -lcrypto`
against the downloaded HDF5 matrix and the prepared replay. The checker reads
the source CSC datasets directly through the HDF5 C API; it does not reuse the
CellShard converter or its verification code. It confirmed source SHA-256
`0668e2bf9ff521a8e5e7aa37717f3780bad84081f26f888df16a9398d5020e76`, every cell
barcode, every peak's source index/ID/name/coordinates, and all 23,970,376 packed
feature-major words. The source contains 61,064,492 positive peak observations.
A copied replay with one bit toggled was rejected at peak 0, cell word 0. This
establishes exact replay encoding against this file and fixture; it does not
validate chromatin-gate biology.

I audited Scheduler job `95f3c36a-ec5a-4585-a1d7-fcde5de893b3`, attempt
`9cd19752-fcf2-4657-8003-b3d506ce34bd`: Scheduler reported success and verified
cleanup, its allocation was exclusive on UUID
`GPU-d745da9c-7649-8334-41f1-483afa6f3206`, and the same job, attempt, UUID,
closure SHA-256, and binary SHA-256 appear across packet evidence, the build
receipt, Scheduler receipt and one settled campaign-ledger row. The charged GPU
time is 21.543 seconds of scheduler job wall time. The racecheck packet itself
took 3.517 seconds and reported zero hazards, errors, or warnings. It used a
standalone STE fixture, so this proves neither combined gate/body training nor
general STE correctness.

## Remaining acceptance gaps

The full study capture must explicitly include the prepared replay payload and
both identity sidecars. `freeze_source.py` excludes paths named `cache` by
default and accepts replay entries as individual files. The campaign owner
confirmed that the eventual full capture will pass `accessibility.cgatac`,
`cells.tsv`, `features.tsv`, and `manifest.json` individually, alongside the
original HDF5 file.

The first synthetic STE packet records an `nvcc --version` string. Complete
replay measurements still need source-bound compiler/toolkit, HDF5/OpenSSL, and
sanitizer identities, as applicable, in addition to binary, input, Scheduler
allocation, and cleanup evidence. The campaign owner has confirmed these will
be recorded for replay work.

This review does not establish the parent campaign's live result at later
revisions or remaining budget. That mutable state is owned by the campaign
controller and its receipts. GPU correctness, the other sanitizer modes, full
condition-to-update correctness, and performance claims remain separate
acceptance requirements.
