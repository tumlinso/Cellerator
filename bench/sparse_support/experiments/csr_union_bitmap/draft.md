# CSR posting union versus per-cell edge bitmap

## Question

For ranked-gene style batches with many selected source genes per cell, can
parallel traversal of the queried role-0 CSR postings set edge-ordinal bits and
compact exact work faster than the current CUDA query kernel's serial
multiway merge? The bitmap is per cell and per edge domain. It does not build
the much larger entity-by-edge bitmap table.

## Candidate mechanisms

1. Current production `cuda_query_reverse_postings_any_v1`, including its q=1
   fast path and current q>1 behavior.
2. CSR union bitmap: query only selected source-gene postings, atomically OR
   each logical edge ordinal into a per-cell E-bit set, scan word popcounts,
   and compact IDs/ordinals in stable order.
3. Count/scan/concatenate: for the restricted case where source postings occupy
   ordered, disjoint stable-ID ranges and each query row is sorted and unique,
   count posting lengths, prefix the counts, and copy postings directly.
   Applicability is checked from the fixture; duplicate terms or arbitrary ID
   order make this baseline inapplicable.
4. A strong parallel full-rescan baseline: build per-cell query-membership
   bitmaps, scan every source edge in parallel to count matches by tile, scan
   tile counts, and compact matching work items. This prevents a pathological
   serial merge from being the sole comparator. Candidate bitmap/full-scan
   output uses stable-ID rank; an independent-ID CPU oracle checks that the
   implementation does not rely on ID equaling ordinal.

The run holds the 1M-edge, 20k-entity, 32-cell frozen topology fixed. Query
lengths are nested at 1, 32, 128, 1024, and 2048 source genes; per-cell queries
extend the fixture's shared cohort of 24 selected genes. A separate duplicate
term control verifies idempotent union behavior. All methods receive identical
query CSR and return exact stable logical IDs/ordinals with explicit overflow
status. The current serial posting merge is timed only through q=128. At q=1024
and q=2048, its `q × eligible-output` work proxy is recorded and timing is
omitted to avoid spending minutes on a known quadratic-in-query-count path;
large-query decisions compare the bitmap with full scan and the applicable
count/scan/concatenate control.

## Cost boundary

Measure host index preparation and device transfer, resident index and scratch
bytes, query construction, per-call zeroing, posting scatter or full scan,
popcount/compaction scans, and complete query cost. Include capacity sizing and
reuse estimates. The frozen fixture has unique directed pairs, source degree
at most 512, one-based IDs, and exact support oracles. The probe also checks a
deterministic independent stable-ID permutation against its CPU rank-bitmap
oracle. Build in `/tmp`, run only through the Device Scheduler on the selected
V100 UUID, and keep all code and evidence under this experiment directory.

The result is a bounded design probe. It does not select a production backend
or establish a universal crossover.
