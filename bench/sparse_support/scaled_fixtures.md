# Scaled sparse-support fixtures

`scaled_fixtures.py` provides deterministic synthetic workloads for support
indexing, bitmap retrieval, gating, and sparse numerical experiments. These
are performance-engineering fixtures; their distributions do not represent
biological measurements. They are generated in memory in linear work with the
number of edges and observations. No pairwise affinity matrix or batch-by-edge
state matrix is materialized.

The small profile is a host oracle with empty and overlapping source queries,
duplicate terms, repeated event occurrences, explicit timestamp/count windows,
cohort-shared query support, independent per-cell values, module-correlated
predicate changes, two predicate participants per edge, and ranked entity IDs
separate from rank positions. The larger profiles include:

| Profile | Edges | Entities | Cells | Purpose |
| --- | ---: | ---: | ---: | --- |
| `large-retrieval` | 1,000,000 | 20,000 | 32 | Multiway role-0 posting union and bitmap comparison |
| `large-indexing` | 1,000,000 | 20,000 | 512 | Shared-topology indexing and cell-cohort scaling |
| `large-indexing-2m` | 2,000,000 | 20,000 | 1,024 | Upper indexing scale |
| `large-ranked` | 262,144 | 4,096 | 32 | Ranked-set workload with degree near 64 |
| `large-events` | 1,280,000 | 20,000 | 8 | Bursty histories and separate time/count filtering |
| `large-numerical` | 262,144 | 8,192 | 128 | Bounded numerical scale input |

Stable `edge_ids` are one-based; edge ordinals and CSR posting values are
zero-based where explicitly named `*_ordinals`. Query CSR uses role 0 (source)
and emits the exact sorted stable edge-ID union as an independent host oracle.
The per-cell `query_offsets/query_terms` contain unique sorted selected source
IDs. The separate `query_indptr/query_entities` cases cover empty, unique, and
duplicate term lists at sizes 1, 32, 128, 1,024, and 2,048 when the entity count
allows them.

Layout controls preserve logical IDs, endpoints in logical order, events,
ranked identities, and query results. `physical_to_logical` is a permutation
from physical slot to logical edge ordinal; `logical_to_physical` is its inverse.
Only endpoint arrays in physical order and the layout tag change. This makes
identity and shuffled storage directly comparable without changing graph
degree or query work.

## Generate and share identical bytes

```sh
python3 bench/sparse_support/scaled_fixtures.py \
  --profile large-retrieval --layout identity \
  --bundle-dir /tmp/cellerator_sparse_retrieval_identity
```

The bundle contains raw little-endian `.bin` arrays and `manifest.json`; each
array entry records the key, dtype, shape, byte count, and SHA-256. Identical
array aliases share one raw file and name the canonical key in `alias_of`. A
second consumer can load the same arrays without parsing expanded edge lists. Use
`--layout shuffled` to generate the physical-layout control. `--metadata-only`
prints shape, degree, query, reuse, gate-change, memory, and topology digest
metadata without writing the arrays.

Run the fixture-only host checks with:

```sh
python3 -m unittest tests.sparse_support.test_scaled_fixtures
```

These checks validate deterministic hashes, sorted stable IDs, query unions,
layout invariance, event-window indexing, gate/predicate structure, cohort
support reuse, independent cell values, rank identity, and raw-bundle hashes.
They do not build production code or run CUDA.

The campaign's frozen seed `20261027`, identity-layout bundle manifests have
these SHA-256 values (the manifest includes each array's dtype, shape, file,
and SHA-256):

| Profile | Shape (E, N, B) | Manifest SHA-256 | Topology SHA-256 |
| --- | --- | --- | --- |
| `small` | (96, 64, 8) | `5d0c2c09ee8f54130e0af4ba4d906b85ee2b013c9ea397a530268381ba742302` | `397b678bb4a7b749751d34c48c01d7a31726438eeffc5679dbd7ea7a529c5c51` |
| `large-ranked` | (262,144, 4,096, 32) | `a4224683b6a04ad533d6841cc24052ccc7ec961718211c1108d207c317c160fa` | `bb636a9bd12b0b4c4e3c12c4a049238004e60a7c1f590d983ea8c22f4643648f` |
| `large-events` | (1,280,000, 20,000, 8) | `bf860b0cc9ca2619316aeeb356f1bdc9b6b68dd65c6b3a7ee9f2c3f9bc3e69c1` | `912a38261ac581f6be1c1db3a2a5c267dbd437f93f2136408190af5c8176bc4a` |

At this freeze, `scaled_fixtures.py` has SHA-256
`fd6e304a2ff48e2cb23255e8fab51253bdc01ff2861c1aa2fa621dc6174cb20e`.
