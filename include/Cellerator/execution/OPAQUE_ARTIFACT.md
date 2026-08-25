# Opaque execution artifact delivery

CellShard owns CSPACK publication, exact-range fetch, host residency, device
placement, and the single caller-stream upload. It validates the CPEXEC01
transport envelope as opaque bytes and does not include or interpret CPE2
directories, projections, or operation semantics.

Cellerator's `validate_opaque_execution_artifact_host` first requires the exact
CellShard transport identity, then validates the inner CPE2 checksum, structure
epoch, semantic geometry, projection catalog, and image identity. The outer
legacy row-domain and feature-axis fields remain explicit compatibility facts;
they are not hashed into or substituted for CPE2's 128-bit identities.

`bind_opaque_execution_artifact_device` accepts an already uploaded CellShard
residency. It checks that the device identity and size match validated host
state, then forms CPE2 device-relative projection pointers from the validated
host directories. It performs no allocation, copy, parsing, checksum,
synchronization, stream selection, or device selection.

The validated host view is cold load state. The bound device view is prepared
state and owns neither the CellShard allocation nor launch values, stream, or
workspace. Host and device residencies must outlive their corresponding views.
Persistent identity and generation checks, never pointer equality, determine
compatibility.
