# Relation Update Spine v1 WMMA policy

`select_gradient_route` is a bounded prepared choice. `automatic` remains
sparse for both arithmetic profiles until whole-lifetime measurements justify a
crossover. `force_sparse` calls the existing complete sparse provider. Forcing
hybrid changes profitability only: it requires explicit half-rounded operands
and a nonempty legal exact cover. An unknown choice, missing cover or full-f32
WMMA request fails with a reason. Preparation and enqueue still validate support,
capacities, alignment, lifetime, stream and protected allocation overlap.

The numerical operation is defined in the architecture spine and
[the RU1 numeric contract](../../planning/relation-update-spine-v1/02_SEMANTIC_AND_NATIVE_CONTRACT.md).
Full-f32 stays outside WMMA. Both rounded realizations consume binary16 RNE
operands widened for accumulation; neither silently changes the full-f32 request.

## Reachable mechanisms and phase evidence

The hybrid extension borrows the native pair's immutable physical support and
whole-operand half scratch. It owns only gathered panels, score scratch,
extraction maps and residual descriptors. Preparation does not create a second
value plane or training engine.

| Selected realization | Actual stages per nonempty invocation |
| --- | --- |
| Full-f32 sparse | Existing full-f32 physical edge dot |
| Half-rounded sparse | Two fresh operand pack kernels, existing complete half sparse dot |
| Half-rounded hybrid | Two whole-operand packs, two gathered panel packs, repaired rectangular WMMA, supported-score extraction, optional disjoint sparse residual |

`pack_refreshes` counts paired refresh epochs; `pack_launches` counts actual
pack kernels. `wmma_launches`, `extraction_launches`, `residual_launches` and
`sparse_launches` identify submitted provider stages. Counters do not claim
asynchronous completion or benchmark promotion. Outputs remain in physical edge
order. Tile holes are never written as biological parameters. No accumulator
fusion with extraction or residual is claimed.

W04 device evidence `1af9d7a2-067a-4eec-9e76-cbd46383cf2e` ran CUDA 12.9 sm70
under the controller's GPU reservation and benchmark mutex, with Compute
Sanitizer memcheck reporting zero errors. Dense 256-edge and mixed 255+6-edge
fixtures each executed two WMMA and two extraction launches across finite and
nonfinite passes; the mixed fixture also executed two residual launches. The
six-edge sparse fixture executed no WMMA. An independent double-summed reference
used explicitly rounded half operands. This is correctness and mechanism
evidence, not a timing comparison or a performance promotion.

## Costs and lifetime

Preparation reports a conservative peak bound for flat lookup tables, host
mapping vectors, device metadata, panels and score scratch. The default bound
is 256 MiB; a caller-supplied positive cap is enforced before corresponding cold
allocations. `persistent_bytes` includes device metadata and the provider object;
`scratch_bytes` includes both gathered half panels and f32 score tiles. The
pair must additionally report its borrowed whole-operand scratch and support.
For T tiles, provider scratch is 2048*T bytes. Empty sparse work launches nothing;
forcing hybrid on a zero-tile cover is explicitly unsupported.

Preparation uploads maps and drains its owner stream before freeing host staging
vectors. Teardown drains that stream. Hot execution performs no allocation,
readback, topology search or canonicalization. Operands refresh every invocation,
including same-address or same-version submissions; there is no pack-cache
validity claim. Borrowed buffers remain valid through asynchronous completion.
The enclosing pair owns the real publication/read-lease semantics.

Any promotion measurement must separately include preparation, operand packing,
WMMA, extraction, residual, update, publication, transfers and required
observations; record persistent and peak scratch bytes and number of repeated
invocations. Compare identical rounded numerical policies. Tiny correctness
fixtures primarily expose launch overhead and cannot establish a speedup.
Integrated I05/I08 device gates and I06 lifetime timings remain separate required
acceptance work; these lane receipts do not replace them.

## DEFER-WMMA-01

Revisit the general contraction/hybrid portfolio, shared panels across
apply/transpose/VJP, alternative tile granularity, wider widths, integrated
candidate costing and alternative authoritative orders before stabilizing these
interfaces. This N16 provider is not a graph-wide planner, arbitrary-N family,
Ampere pipeline, new precision portfolio or universal 16x16 geometry policy.
