# Qualification evidence

The parent [implementation record](../IMPLEMENTATION.md) states the bounded
claims. `manifest.json` identifies original receipts, a lossless archive of raw
scale numerical results, host logs and earlier attempts. Host `.log.gz` files
decompress to their exact captured bytes. `gpu/frozen-binaries.json`
records source hashes, compiler/link flags and binary hashes. Only the recorded
operations are runtime-qualified; linked dependency closure is build provenance.

`scale/numerical-results.tar.gz` preserves the exact raw and wrapped numerical
JSONs, comparisons and campaign. Its members and byte hashes are in the
manifest. Extract it into an empty temporary directory for independent replay
with `bench/native_data/compare_backends.py`; the archive has relative paths
and regular files only. The original immutable Autobench artifacts remain in
the configured local evidence store.

GPU jobs require authenticated Device Scheduler admission. The copied specs
and frozen driver record the original commands/paths; those paths do not
constitute a portable allocation grant. Rebuild the current CMake targets for
a new run, record new binary identities, and submit a bounded job. Use the
CUDA12.9 Compute Sanitizer executable directly: the NVHPC dispatcher selected
a missing13.1 installation during an earlier attempt.

`gpu/peer_access_already_enabled.xml` matches only the explicitly handled
`cudaGetLastError`/`cudaErrorPeerAccessAlreadyEnabled`/704 result in the
prepared-transfer constructor call chain. Its leading empty frame corresponds
to the generated libcuda frame. The captured canonical XML is retained beside
it; human-readable backtraces resolve the anonymous helper as
`select_peer_path`. Memory/leak checks and all other API errors stay enabled.
No unsuppressed/tool-setup attempt is represented as a clean result.

Timing has no samples. All three bounded attempts were rejected before worker
execution; `timing/attempts.json` and its raw receipts record this. The workspace
was restored to paused after all attempts became terminal. No resource control
or numerical acceptance criterion was bypassed.
