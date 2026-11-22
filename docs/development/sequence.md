# Native sequence

Use `Cellerator::sequence_ops` for host packing, exact queries, and predicate
operations. The same component is selectable in the Substrate SDK. A CUDA build
also provides `Cellerator::sequence_ops_cuda` for stream-ordered packing, masks,
count, and emit.

## Source ownership

| Responsibility | Header path under `include/Cellerator` |
| --- | --- |
| Nucleotides, strand, source coordinates, and halo ownership | `biology/sequence.hh` |
| Borrowed two-bit words and split bit-plane views | `tensor/sequence/dna2.hh` |
| Pack into supplied words, validity masks, or split planes | `layout/sequence` |
| Exact queries, motif operations, events, and bounded predicate preparation | `ops/sequence` |
| Bind sequence views and prepare or run sequence-to-regulatory execution | `execution/sequence` |

A source chunk names a genome domain, contig, and origin. The owned span controls
which anchors can emit output. A window can read valid halo bases across a
chunk boundary. Reverse traversal keeps the original global coordinates.
Validity distinguishes a known nucleotide from an ambiguous source position
and from padding beyond the logical extent.

Provision words, masks, event records, and result counters before calling the
operation. Keep borrowed storage live through its final consumer. The
regulatory binding borrows existing split planes with explicit validity for
nonempty inputs, and can use them across cell contexts. Its prepared CUDA path
accepts one forward exact-motif mask and supports fused, mask-materialized, and
relation-materialized execution. Cache reuse depends on source generation,
coordinate identity and epoch, and, for materialized relations, projection
identity.

Use the focused checks in `tests/sequence` and `tests/execution/sequence` for
exact operations and regulatory parity. Run the sequence benchmarks under
`bench/sequence` to account separately for packing, transfer, and resident
repeated execution. The [source history](../history/baseplane.md) retains the
ordered hierarchy, edit repair, and learned routing experiments.
