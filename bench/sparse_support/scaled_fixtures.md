# Scaled sparse-support fixtures

`scaled_fixtures.py` provides deterministic synthetic workloads for support
indexing, bitmap retrieval, gating, and sparse numerical experiments. These
are performance-engineering fixtures; their distributions do not represent
biological measurements. They are generated in memory in linear work with the
number of edges and observations. No pairwise affinity matrix or batch-by-edge
state matrix is materialized.

The small profile is a host oracle with empty and overlapping source queries,
duplicate terms, repeated event occurrences, explicit timestamp/count windows,
cohort-shared query support, independent per-cell values, gene-local and
cohort-correlated latent predicate changes, two predicate participants per
edge, and ranked entity IDs separate from rank positions.

Every generated source/destination pair is unique and non-self. Outdegree is
Zipf-like and capped at `min(512, N-1)`; edge neighborhoods take as many module
local destinations as the configured local fraction and fill the remainder
from other modules. This preserves skew and shared local topology without
letting a posting's degree exceed the entity universe. Query candidates are
exact unions over stable edge IDs. Their recorded counts are bounded by
`query_terms * max_source_degree`.

Each edge has exactly two conditions: its observed source-gene predicate ID
(`0..N-1`) and a module-shared latent predicate ID (`N + module_id`). Per-cell
observed predicate values are sparse CSR over selected genes. Local gene flips
record `(cell, gene, timestamp, new_value)`; broader latent flips record
`(cell, module, timestamp, new_value, cohort_group)`. Exact edge fanout for
each change class is included in the arrays and metadata. No pretrained
chromatin or biological ontology is encoded.

Observed views have explicit labels: `0` is generic selected-gene state, `1`
is ranked-set presence, and `2` is event-window history. Ranked presence uses
`ranked_presence_offsets/entity_ids/values`; event-window observations use
`event_window_observed_indptr/gene_ids/edge_ids/values`. These views remain
separate, so the 24-gene generic query state does not silently gate ranked or
event observations. `large-ranked` uses 512 ranked entities per cell; 75% of
each cohort's identities are shared, with an independently shuffled rank order
and cell-specific residual for each member.

Per-cell histories retain their requested length. Bursts repeat selected
source-support edges, while fresh occurrences mix selected-support edges with
background edges. Timestamps increase within each cell, count and time windows
are separate bounds, and `event_window_query_*` gives the exact selected-gene
intersection for every window.

The larger profiles include:

| Profile | Edges | Entities | Cells | Purpose |
| --- | ---: | ---: | ---: | --- |
| `large-retrieval` | 1,000,000 | 20,000 | 32 | Multiway role-0 posting union and bitmap comparison |
| `large-indexing` | 1,000,000 | 20,000 | 512 | Shared-topology indexing and cell-cohort scaling |
| `large-indexing-2m` | 2,000,000 | 20,000 | 1,024 | Upper indexing scale |
| `large-ranked` | 262,144 | 4,096 | 32 | 512 ranked identities per cell with cohort overlap |
| `large-events` | 1,280,000 | 20,000 | 8 | Bursty histories and separate time/count filtering |
| `large-numerical` | 262,144 | 8,192 | 128 | Bounded numerical scale input |

Stable `edge_ids` are one-based; edge ordinals and CSR posting values are
zero-based where explicitly named `*_ordinals`. Query CSR uses role 0 (source)
and emits the exact sorted stable edge-ID union as an independent host oracle.
`source_entity_id` and `participant_entity_id` are zero-based entity IDs.
`cell_query_indptr/cell_query_entities` (also `query_offsets/query_terms`)
contain unique sorted per-cell source IDs. `query_indptr/query_entities` cover
empty, unique, and duplicate term lists at sizes 1, 8, 32, 128, 1,024, and
2,048 when the entity count allows them. `source_degree_histogram_edges` and
`source_degree_histogram_counts` report the exact source-degree histogram.
`event_window_query_indptr/event_window_query_indices` contain the exact
intersection of each event window with that cell's selected source genes.
`array_bytes` counts unique in-memory arrays once, even when several schema
keys alias the same NumPy array.

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

These checks validate deterministic hashes, unique endpoint pairs, the degree
cap and histogram, exact query unions/cardinality, layout invariance,
event-window indexing, event/window/query intersections, gene-local and latent
predicate structure and fanout, cohort support reuse, independent cell values,
rank identity, and raw-bundle hashes.
They do not build production code or run CUDA.

The earlier `fd6` manifests remain immutable diagnostic data; they contained
repeated endpoint pairs and were not used for performance qualification. The
replacement freeze uses seed `20261027`, identity layout, and the following
hashes. Each manifest lists every array's dtype, shape, file, and SHA-256:

| Profile | Shape (E, N, B) | Manifest SHA-256 | Topology SHA-256 |
| --- | --- | --- | --- |
| `small` | (96, 64, 8) | `c33facc18f330739411c1f293609c237689685bd4003099ca79b331fded144be` | `3755cf50273d5a8012cf036058f8c3ab77ded7bcc72f20254369e775d10b3b3b` |
| `large-ranked` | (262,144, 4,096, 32) | `674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32` | `2cadfde9b62b6e9bfc49cce9f44219e241fec21e7c68dfe685d90fe372aeabc5` |
| `large-events` | (1,280,000, 20,000, 8) | `910a4c742b795a2a827cb21e62c510a3bc5039e6878ae75d4826fc0d33234be8` | `4d34d686656c0aa4bb1d89368993d9d31476ca568cd8b5d87f8231783efa9a38` |

At this freeze, `scaled_fixtures.py` has SHA-256
`887437fc3418e06c25e515487891a3224f8dd56493b689bf41bc70e66ae5a76b`.
