# Sparse-support evaluation intake

This CPU-only front door accepts data that has already been tokenized. It
preserves supplied identities and order, builds exact sparse candidate views,
and writes the existing `cellerator.scaled_sparse_fixture_bundle.v1` raw-bundle
format. It does not infer biology, normalize data, fit statistics, or download
datasets. NumPy is the only runtime dependency; it does not import Torch or
the CUDA extension.

## Retained mechanism catalog

Run `python -m bench.sparse_support.evaluation catalog` from the repository
root. The JSON catalog in
[`families.json`](../../bench/sparse_support/evaluation/families.json) lists
each mechanism's source, evidence, status, and measured scope. The catalog
retains native reverse/temporal indexes, conditional worksets, bounded support
proposals and support deltas, learned latent gates, sparse SIMT execution,
CE-GEO N64 execution, indexed shared edge values, CSR and bitmap retrieval,
ordered posting concatenation, counted flat conditions, shared pair-condition
caching, and the parallel workset probe. Experimental entries remain available even
when synthetic or fixture evidence shows a workload-specific loss. GPU-measured
means only that the referenced workload was run; it does not claim a universal
winner. CE-GEO N64 width 64 is a physical consumer anchor, not a logical input
width restriction.

## Python API

The import path is `bench.sparse_support.evaluation`:

```python
from bench.sparse_support.evaluation import (
    CapacityLimits, prepare_events, prepare_ranked, write_raw_bundle,
)

ranked = prepare_ranked(
    edge_ids=edge_ids, source_entity_ids=regulators,
    destination_entity_ids=targets, cell_ids=cell_ids,
    gene_ids=already_tokenized_genes, rank_positions=rank_positions,
    token_ids=token_occurrence_ids, provenance={"dataset": "local-cohort-v1"},
    capacities=CapacityLimits(edges=2_000_000, query_records=10_000_000),
)
write_raw_bundle(ranked, "results/ranked-input")

events = prepare_events(
    edge_ids=edge_ids, source_entity_ids=regulators,
    destination_entity_ids=targets, cell_ids=event_cells,
    entity_ids=positive_event_genes, occurrence_ids=occurrence_ids,
    event_ordinals=caller_ordinals, timestamps=event_timestamps,
    position_windows={cell_id: (first, stop)},
    timestamp_windows={cell_id: (time_lo, time_hi)},
    provenance={"source": "local-event-export"},
)
write_raw_bundle(events, "results/event-input")
```

Topology is a directed relation with opaque signed 64-bit entity, edge,
occurrence, and token IDs. Zero and negative external IDs are valid. Duplicate
edge IDs and token/occurrence IDs are rejected. Source and destination posting
lists are exact and sorted by logical edge order. Candidate output is CSR by query record; `candidate_offsets`
and `candidate_edge_ids` preserve variable degree without a dense cell-by-edge
allocation. `query_role` can be `source`, `destination`, or `either`; the latter
unions and deduplicates the two relation roles.

The event arrays are parallel flat arrays. Records are grouped by cell in the
output while maintaining caller order within each cell; `*_input_indices` and
`query_record_ids` preserve the original record identity. `event_ordinals` are
opaque fields and may repeat or be nonmonotonic. `position_windows` use
half-open positions in each cell's caller-provided event order. Timestamp
windows independently filter supplied timestamps and never reorder records.
Ranked inputs retain gene identity and rank independently; the ranked API has
no temporal-window parameter.

The native ABI uses positive `uint64` identities and requires dense edge
ordinals; the temporal ring also requires positive cell/entity/occurrence IDs
and monotonically increasing event ordinals. The intake preserves all external
IDs and ordinal values in the ordinary fields, then emits explicit parallel
`native_*` arrays and reverse maps (`native_entity_external_ids`,
`native_cell_external_ids`, and aligned `edge_ids`/`native_edge_ids`) assigning
positive dense identities in canonical sorted order. Candidate edge IDs have
`native_candidate_edge_ids` aligned with candidate ordinals;
`native_edge_ordinals` are the dense zero-based positions into canonical edge
arrays. `native_query_record_ids` is aligned with external `query_record_ids`
(event occurrence IDs or ranked token IDs), and `native_query_entity_ids` and
`native_query_cell_ids` are aligned with their external query arrays. Event
`native_event_ordinals` is a per-cell ordinal derived from caller sequence
position, while the caller's possibly nonmonotonic `event_ordinals` remains
unchanged. These maps make the conversion reversible and visible; downstream
adapters must use them deliberately. Timestamps remain as provided: because the
native temporal ABI takes integer timestamps, a consumer must choose and record
an explicit time scale before using timestamp windows natively.
These maps do not supply relation IDs, structure epochs, participant role
labels, or a native support-view object; an execution adapter must bind those
from its own contract. A raw bundle can therefore be intake-valid without
being immediately executable by a native backend.

The dense maps describe one prepared bundle. An adaptive execution adapter
must retain and extend an established mapping, or create a distinct native
identity namespace when rebuilding it. Recomputing sorted dense IDs after
adding an external identity can otherwise reassign existing native IDs. The
intake does not publish support revisions or replace the library's stable
logical identities.

Condition IDs and participant IDs are opaque caller-provided arrays. Each edge
may reference zero or more condition IDs; each cell may supply active condition
IDs. When `active_conditions` is supplied, eligibility requires every condition
attached to an edge to be active in that cell. This is a generic conjunction,
not an inferred condition expression or a biological interpretation. The
bundle stores condition participants, edge-condition CSR, active-condition
CSR, structural candidates, and condition-eligible candidates separately.

`CapacityLimits` may bound edge count, query-record count, candidate items per
view, and total declared condition participants. An exceeded bound raises
`CapacityExceeded` with resource/capacity/required-count details; it never
silently truncates. No limits are imposed unless supplied. Arrays are sparse in
the number of records and support incidences rather than `cells × edges`.

## Raw-bundle/runner adapter

`write_raw_bundle(bundle, directory)` refuses a nonempty destination and writes
`manifest.json` plus one raw file per array. The manifest uses the same schema,
descriptor fields (`file`, `dtype`, `shape`, `nbytes`, `sha256`), and hash checks
as `bench/sparse_support/scaled_fixtures.py`. `load_raw_bundle` verifies it;
existing `bench/sparse_support/packed_training_benchmark.py::load_raw_bundle`
also reads the output's manifest and selected raw arrays. This is schema
compatibility only: existing KDA driver cases are not automatically wired to
these mode-specific fields, native ID maps, condition definitions, or windows;
a runner adapter must select arrays and supply its own relation/epoch/backend
metadata. The `metadata.evaluation` section records input mode,
provenance, window semantics, summary counts, and the fact that no dense matrix
was materialized. The normalized arrays and input-index maps are available to
downstream KDA specifications. A runner adapter can consume the checked arrays
and select only the fields its admitted backend needs.

The adapter uses the established sparse-fixture field names `source_entity_id`
and `participant_entity_id` for directed source and destination endpoints;
`edge_input_index` maps the canonical logical-edge order back to caller order.
Query records use `query_*` fields, while the mode-specific `ranked_*` or
`event_*` fields carry token/occurrence identity and original input indices.

CLI intake supports `.npz` arrays with `allow_pickle=False` and a JSON spec for
opaque mappings and provenance:

```sh
python -m bench.sparse_support.evaluation catalog
python -m bench.sparse_support.evaluation prepare-ranked \
  --input-npz ranked_arrays.npz --spec-json ranked_spec.json \
  --output bundle/ranked-cohort
python -m bench.sparse_support.evaluation prepare-events \
  --input-npz event_arrays.npz --spec-json event_spec.json \
  --output bundle/event-cohort
```

NPZ topology keys are `edge_ids`, `source_entity_ids`, and
`destination_entity_ids`. Ranked record keys are `cell_ids`, `gene_ids`,
`rank_positions`, with optional `token_ids`, `cell_id_universe`, and `targets`.
Event record keys are `cell_ids`, `entity_ids`, `occurrence_ids`, and
`event_ordinals`, with optional `timestamps`, `event_values`,
`cell_id_universe`, and `targets`. The JSON spec accepts `query_role`,
`conditions`, `edge_conditions`, `active_conditions`, `target_kind`,
`provenance`, `capacities`, plus `position_windows` and `timestamp_windows`
for event mode. All condition/window mapping keys are cell/condition/edge IDs
encoded as JSON object keys.

## Evidence boundary

The intake implementation and tests establish input integrity, exact CPU
candidate retrieval, sparse storage shape, and raw-bundle compatibility. They
do not establish biological validity, learned-support quality, GPU performance,
or suitability of any one native backend for a caller's data. Use the catalog's
evidence links and a separately admitted KDA experiment for backend
qualification; preserve both wins and negative results by workload.
