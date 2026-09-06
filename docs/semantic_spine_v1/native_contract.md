# Internal native relation pair contract

This is the bounded Semantic Spine v1 interface, not an installed SDK promise.
Mathematical validity and provider capability are separate. Required execution
is N1, binary16 relation storage and float32 input/multiply/accumulation/output,
round-to-nearest, permitted FMA/reassociation, propagated nonfinite values,
overwrite and prohibited aliasing. Reject every unsupported policy explicitly.
No claim is made that a device nonfinite scan occurred.

## Preparation

`prepare_relation_pair` requires a nonnull output slot whose value is null.
It sets that slot null before other validation; callers must not pass an owned
handle there. Success transfers one opaque owning pair to the caller. Forward
and transpose descriptors differ only in direction and bind the same topology,
arithmetic, width, update and alias rules. The pair fixes one device and caller
stream, and owns its projection/value capacity. No second runtime session or
host math backend is introduced.

The CSR view has destination rows and source columns, with every entry in the
declared logical edge order. Validate row count plus one without overflow,
initial offset zero, monotonic offsets, terminal edge count, capacities, source
bounds and provider local-index limits before narrowing. Logical64 metadata is
not a license to truncate into uint32. Duplicate endpoints retain separate edge
identities and additive contributions; a limited provider must explicitly reject
them. Empty support is valid, with explicit device zero-fill/no-op handling.
Host arrays are consumed during the call: asynchronous uploads must own retained
staging or complete before returning. No later launch borrows caller host CSR.
Allocation/upload failure releases partial ownership and leaves the output null.

## Publication and launch

`publish_values` validates device, stream, structure, epoch, logical edge order,
count and strictly increasing nonzero generation. Equal/older generations are
stale errors. The caller retains its device values until stream completion.
Packing uses the pair's fixed buffers on that stream. After successful submission
only, inspection reports the latest **enqueued** generation and increments the
refresh counter. It does not assert device completion. Empty support still
requires a valid generation and metadata, but no inaccessible zero-length data.

`enqueue` validates the entire descriptor against the prepared directional
contract, exact axis identity/order/extent, required count, device, stream,
expected generation and prohibited input/output overlap before any submission.
Pointer rebinding is allowed; partial byte-range overlap must also be rejected.
The caller owns input and output storage through stream completion. Empty output
requires no pointer; nonempty overwrite output must be addressable even when
support is empty. A mathematical alias permission is not provider support.

Metadata rejection preserves output, published generation and accepted launch
counters. If a CUDA error follows partial submission, output/storage may have
changed: poison the pair, report cuda_failure and reject subsequent publication
or enqueue with invalid_state. Do not report a partially packed generation as
accepted. CUDA asynchronous failures still require ordinary caller stream error
observation; counters are submission evidence, not completion evidence.

Repeated publication/enqueue may not allocate, discover geometry, compute host
math, hash support, silently canonicalize or globally synchronize. The declared
orders must be honored by selected projections and explicit preparation maps.
Pair calls require external host serialization, including inspect and destroy.

## Inspection and destruction

`inspect` rejects a null report and otherwise reports actual bound candidate and
projection identities, topology preparation count, successful refresh count,
accepted directional submissions and latest enqueued generation. Candidate
strings remain pair-owned/static and valid through pair destruction. It does
not synchronize or manufacture execution evidence.

`destroy(nullptr)` is harmless. Destroy may fence this pair's stream so its
buffers/staging can be released safely, including after submission failure. The
caller must keep the stream alive through destruction. It does not own caller
buffers or the stream. No multi-device/multi-stream concurrent protocol is implied.

C03 validation: normal supplied demo compiled with g++ -std=c++17 -fsyntax-only
-Iinclude -I/usr/local/cuda-12.9/include examples/semantic_spine_v1/regulatory_reuse.cc
(exit 0). Semantic header standalone C01/C02 host compilation requires no CUDA.
This checks declarations only; GPU execution remains pending the NATIVE lane.
