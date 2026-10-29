# Exact-revision CUDA parameter transfer

Task-local KDA draft, 2026-10-29. KDA entrypoint/starter/flow read at revision
277c79b; CUDA skill host-device pipeline and native-system guidance consulted.

Baseline: native conditional ParameterStore has immutable host revisions and
RAII leases; CUDA runtime has Stream/Event/DeviceBuffer/PinnedBuffer but no
connection between these owners. Implement byte-exact staged H2D transfer,
without numeric conversion, cache selection, GPU execution or benchmarking.

Plan: validate an exact host ParameterLease before CUDA activity; account equal
pinned-host and device byte reservations; allocate staging/device storage; copy
host bytes into owned pinned staging; enqueue H2D and record a readiness event.
Shared transfer state retains the exact source lease. A separate execution lease
inserts a stream wait, exposes the device view and records downstream completion.
Last-owner destruction drains outstanding work before releasing buffers.

Risks: exception between enqueue and event record; early destruction of pinned
host storage; early release after a consumer kernel; mismatched/replaced logical
revision; unowned borrowed streams. Use synchronous drain on exceptional paths
and fail closed if destructor drain cannot establish completion. Borrowed stream
must survive until its completion event is recorded.

Validation: compile source and smoke with nvcc -std=c++17 -arch=sm_70, link native
conditional and CUDA runtime; run smoke --host-only only. GPU smoke compares
downloaded bytes with independent host bytes and checks revision retention but
is supplied unexecuted. No performance claims or candidate comparison. Source
handoff by18:45 UTC; integration owner owns CMake registration.
