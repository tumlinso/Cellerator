# CE-LIVE native training executable v1

`training_program` is the execution-layer wrapper for the deliberately bounded
N=16 native training slice. It prepares the immutable FMP1 forward and CTP1
transpose projections once, retains the sole Cellerator execution session, and
keeps every mutable binding in `training_program_launch`.

The enqueued step is:

```text
FMP1 forward -> bias/ReLU/RMS epilogue -> CTP1 backward
             -> sparse-value SGD -> bias SGD -> generation publication
```

This is not a general-N training engine. The public compile request rejects any
width other than 16 and verifies that the forward and transpose projections
share structure identity, epoch, forward projection identity, runtime handles,
shape, and logical edge count.

## Prepared and launch state

Prepared state contains only the typed non-owning projections, biological axis
identities, device ordinal, explicit output-order contracts, backend identity,
session reference, and counters. Input/output/value/bias pointers, optimizer
scalars, caller stream, workspace, expected generation, next generation, and
readiness records remain launch state. Changing compatible pointers or moving
from one value generation to the next therefore does not rebuild topology.

The run path waits for a non-initial input generation using the runtime event
contract, enqueues all five native stages on the caller stream, and publishes
the next generation only after enqueue status is known. Same-stream waiting is
a no-op; cross-stream visibility is an event wait. It performs no allocation,
device selection, descriptor creation, structural hashing, host wait, or
device-wide synchronization.

The result reports the actual native backend, FMP1 and CTP1 projection
identities, preserved module and feature output orders, structure epoch,
consumed and published generations, completion stream, readiness record, and
native relation-value and dense-bias parameter descriptors.

## Correctness and conventional baseline

`tests/execution/training_program_test.cu` covers one prepared topology across
two value generations and relocated dense pointers. It independently checks
the forward values, explicit parameter/result metadata, width rejection,
transpose/forward identity rejection, missing or stale readiness, insufficient
workspace, and absence of structural re-preparation.

The underlying slice remains covered by
`celleratorNativeTrainingSliceTest`, including independent CPU forward,
backward, sparse-update, and bias-update references plus a persistent generic
CSR forward/transpose baseline with separate epilogues. The retained V100
evidence measured the complete N=16 native step at 19.584 us median and the
complete persistent conventional path at 28.352 us under the same preparation
exclusion, warmup, repeats, and correctness tolerances. This historical
comparison motivates the bounded native backend; it is not generalized beyond
the recorded small-module regime.

The machine-readable provenance for that comparison is retained under
`bench/ce_live/training/native_training_v1_evidence.json`.

## Boundary

- Topology is immutable; learned sparse values and bias are mutable parameters.
- Forward remains feature/gene source to module/row destination.
- Backward uses CTP1's explicit forward-value-position map and does not reverse
  logical edge identity.
- Runtime readiness is not persistent identity.
- CPE2 remains pointer-free and typed projection activation remains non-owning.
- The existing execution session remains the only runtime resource owner.
- No Torch dependency or framework-owned parameter storage is introduced.
