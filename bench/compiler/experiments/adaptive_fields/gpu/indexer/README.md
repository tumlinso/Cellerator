# E7 scan versus ordered-posting bridge

This experiment-local C ABI exposes two exact candidate-discovery paths over
the same prepared logical support and query CSR:

* `AF_INDEX_SCAN_V1` marks each cell/edge pair against that cell's query terms,
  performs a device prefix scan, and materializes a CSR workset.
* `AF_INDEX_ORDERED_POSTINGS_V1` calls Cellerator's existing
  `cuda_query_reverse_postings_any_v1`. Its device implementation proves the
  ordered-disjoint concatenation precondition per row and otherwise invokes its
  exact stable-ID merge fallback.

Both return stable logical edge IDs, canonical support ordinals, cell offsets,
and flattened gate indices. The gate index for each flattened item is its
position `0..item_count-1`, matching the candidate score vector expected by the
Torch sparse support API. Both are tagged with the structure epoch from
preparation and the caller's state generation. The C ABI accepts zero-based
source/destination endpoints and converts them to Cellerator's one-based
support entity IDs; query terms use one-based IDs and role 0 for source or 1
for destination.

For correctness qualification, call both modes with `verify=1`. The bridge
compares every result ID and ordinal with an independent host scan and reports
ordered-fast-path and exact-fallback row counts. This adds CPU oracle and
precondition-classification work. Timed calls should use `verify=0`; the
returned correctness diagnostic counts are then marked invalid and zero.

## Build and C ABI

From the repository root, the CPU-only build command is:

```sh
python bench/compiler/experiments/adaptive_fields/gpu/indexer/build_bridge.py \
  --output-dir bench/compiler/experiments/adaptive_fields/gpu/indexer/artifacts \
  --arch sm_70
```

The builder compiles the local `.cu` bridge and existing
`cuda_postings.cu`, `reverse_postings.cc`, and `logical_support.cc`, then links
`libadaptive_fields_indexer.so`. The manifest records each input hash,
compiler/toolchain identities, exact commands, and output hash. It performs no
device initialization or CUDA operation. Importing the ctypes facade is also
CPU-only; constructing `Indexer` or calling `.query()` invokes the C ABI and
requires the parent-owned scheduler job.

The ABI is declared in `indexer_bridge.h` and bound in `ctypes_binding.py`:

```python
from ctypes_binding import Indexer, ORDERED_POSTINGS

with Indexer(library, edge_ids, sources, destinations, entity_count, epoch) as index:
    candidates = index.query(
        ORDERED_POSTINGS, query_offsets, (role_entity_terms),
        state_generation=generation, verify=False)
    derivative = candidates.to_torch_workset(torch, device)
```

The caller supplies a host result capacity of `E * cells`, which is a safe
upper bound. `af_index_run_v1` synchronizes the selected stream and copies the
exact CSR rows to those host buffers. `to_torch_workset` constructs host-backed
Torch tensors on the requested CUDA device, so the D2H, Python materialization,
and H2D costs belong inside E7 complete-step timing. This adapter intentionally
does not add a DLPack or zero-copy bridge.

## Workset identities and exact fallback

The support handle is immutable for its structure epoch. A structural update
requires a newly prepared handle. Each query call carries its own state
generation. The output IDs and ordinals are exact and sorted by logical ID
within each cell. Repeated query terms do not duplicate support interactions;
they simply qualify the same edge. Unsorted or duplicate query terms make the
ordered-disjoint proof fail and are handled by the production merge fallback.
Out-of-range cell offsets, nonmonotonic query partitions, stale support, output
overflow, and native status errors are surfaced instead of truncating worksets.

The host fast-path counter is computed only in correctness mode because it is
instrumentation. The production posting kernel does not currently expose a
per-row fallback counter, so timed runs must not infer that counter from host
work not performed on the timed path. Exact output comparisons still qualify
the two modes.

## Current boundary

The ABI was compiled and its exported symbols were checked on the CPU. GPU
array parity, stream behavior, capacity on scaled workloads, and complete
training-loop timing remain unvalidated until an admitted Device Scheduler
window. No production file, API, kernel, or default was changed.
