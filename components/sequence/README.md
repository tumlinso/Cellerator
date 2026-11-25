# Cellerator sequence

[Baseplane's scientific charter](https://github.com/tumlinso/Baseplane/blob/8edb8e5ed67c5ba522b4e33952becdc843f08c68/docs/FOUNDATIONS.md) sets out to learn hierarchical floating-point representations of
genome-wide information while retaining a path back to exact bases, strand,
validity, and source coordinates. The encoder should discover useful
organization as it works. Nearby sequence is cheap to process together, and
learned relationships can bring distant loci together when a question requires
them. A summary can request finer detail from its exact source.

This objective complements [Cellerator's cellular geometry](../../cellular_geometry.md).
Sequence provides facts, candidate relationships, and features with provenance.
A cellular model expresses admissible dependencies and the state that conditions
them. Its state can also guide which sequence regions need more detail. The
computational hypothesis is that useful organization can reduce total work,
including discovery, routing, movement, and source revisits.

## Current capabilities

The component provides the bounded exact-sequence foundation: source chunks,
borrowed two-bit words and split bit planes, packing, motif scans and masks,
exact-source queries, event records, and predicate preparation. The learned
genome-wide representation remains a research objective. The
[Baseplane source history](../../docs/history/baseplane.md) links the original
source and experiments in hierarchy, edit repair, and learned routing.

## Use

Request `find_package(Cellerator CONFIG REQUIRED COMPONENTS sequence)` and link
`Cellerator::sequence`. Use public headers under `Cellerator/sequence/` or
`import cellerator.sequence;` when configured with
`CELLERATOR_ENABLE_MODULES=ON`. The API is in `cellerator::sequence`.

The normal Cellerator build includes the stream-ordered CUDA operations when
CUDA is enabled. CUDA callers include `<Cellerator/sequence/cuda.hh>` and link
the same target. The selectable Substrate SDK builds the host component
with `CELLERATOR_SUBSTRATE_COMPONENTS=sequence`.

A source chunk names a genome domain, contig, and origin. The owned span
controls which anchors can emit output, and a window can read valid halo
bases across a chunk boundary. Reverse traversal preserves original global
coordinates. Validity distinguishes known bases, ambiguous input, and padding.

Provision packed words, masks, event records, and counters before calling
an operation. Keep borrowed storage live through its final consumer. The
[internal regulatory executor](../../include/Cellerator/execution/sequence/README.md)
consumes sequence planes and selects among fused, mask-materialized, and
relation-materialized execution for a forward exact-motif predicate.
