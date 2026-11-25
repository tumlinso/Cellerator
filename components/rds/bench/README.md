# RDS storage comparison

Build the `cellerator_rds_access_bench` Release target. Pass the original RDS and
its byte-identical NVMe copy by absolute path. Use a distinct index for each
input and keep all indexes on the same NVMe filesystem. Record both file
digests and device numbers before measurement.

| Operation | Request |
| --- | --- |
| `build SOURCE INDEX CACHE` | Create a new index at an unused path. |
| `reopen SOURCE INDEX CACHE` | Open the source with an existing index. |
| `sweep SOURCE INDEX CACHE WORKERS` | Read all cells in batches of up to 4,096. |
| `scatter SOURCE INDEX CACHE WORKERS` | Read 64 deterministic, shuffled, non-overlapping ranges of 512 cells each. |
| `verify SOURCE INDEX as-is ORACLE_MTX` | Compare the 64 scatter ranges and one combined late/early/middle/repeated request to the column-ordered integer Matrix Market file. |

`WORKERS` is `1` or `4`. Restrict the process to four distinct physical cores on
one NUMA node. The driver pins each worker to one allowed CPU and gives it a
contiguous segment of the same request list. All workers provision one buffer
of identical capacity before timing starts. The driver reports setup, cache
preparation, and transfer intervals separately.

`CACHE` is `cold`, `warm`, or `as-is`. The driver requests file-local page
eviction for cold trials and loads the source into page cache for warm trials.
It measures host page residency immediately before starting timed work.
A verified cold trial has at most 1% of source pages resident. A verified warm
trial has at least 99% resident. Index files and hardware caches are outside
this classification.

Run each finite timing packet through Autobench and retain its JSON output.
Pass the binary and data paths where the worker can see the intended mounts.
Alternate the backing devices for matching cases and use the same CPU list,
binary, and request streams for both. The driver holds the repository benchmark
mutex for timing operations.

Useful bytes count the written values, indices, and offset vectors. Compressed
and decoded byte counters measure RDS payload work in the reader. Process peak
RSS includes worker contexts and output buffers. For index reopen, the exposed
payload counter is zero and the index size is reported separately.
