# CE shared foundation and preservation decision

Source base: `41fcd6e7`. This ADOPT handoff adds the source-tree target
`Cellerator::host_operation_contracts` and
`<Cellerator/compute/operation/host_contracts.hh>`. Include
`host_contracts.cmake` from a child CMake project and link the target to obtain
existing host description headers and their C++20 requirement. It owns no
mutable values, allocator, runtime, numerical implementation or registry.
Consumers executing operations link their explicit implementation owner.

The header exposes existing identity, nf1 contract, indexed incidence, local
arithmetic and relation semantics together without new types. It compiles with
ordinary C++20 and no CUDA headers. The native H01 and N01 consumers execute
actual incidence and host relation implementations and independently check
ordered/repeated arguments, output effects, identity/order mismatches, alias
rejection, widths and numerical policies. The contract consumer checks native
identity/capability seams and rejects an invalid descriptor through its owner.

## Parallel leaves and owner decisions

| Leaf | Entry contract, implementation, consumer | Destination and decision |
| --- | --- | --- |
| STATE | `include/Cellerator/execution/identity.hh`; `experiments/moonshot-parallel-v1/state/state_view.hh`; `state/test_state_host.cc` | New `include/Cellerator/state`, `src/state`; adapt actor/private/ragged views with existing identity, epoch and generation owners. Preserve readout/support/incarnation distinctions. |
| PACK | `include/Cellerator/compute/operation/prepared_relation.hh`; `src/compute/operation/native_numeric/host_relation.cc`; `tests/native_foundation/numeric/n01.cc` | New `include/Cellerator/packing`, `src/packing/core`; strategy/realization wraps cold preparation, retaining immutable geometry and independent value owners. Generic occupancy and frozen Cellpack remain distinct adapters. |
| OPS | `include/Cellerator/compute/operation/native_foundation_contract.hh`; `src/compute/operation/indexed_mechanism/incidence.cc`; `tests/native_foundation/nary/h01.cc` | New matrix/process surfaces; retain ordered arguments and explicit output assembly. Adapt both moonshots, reuse native product2, preserve declared shape/precision/derivative boundaries. |
| EFFECTS | `experiments/baseplane_moonshot/families/effects/include/ce_moon/effects.hpp`; `families/effects/smoke.cpp`; `families/mechanisms/probe.cpp` | New effects surfaces; integrate finite/affine/jet/port algebra with explicit composition order, exactness/truncation and axes. Baseplane keeps sequence provenance. |
| EXEC | `include/Cellerator/compute/operation/indexed_mechanism/training.hh`; `src/compute/operation/indexed_mechanism/training.cu`; CelleraTorch mechanism consumers | Existing execution/native owner paths; preserve one canonical parameter owner, stream-ordered readers, generation publication and saved tapes. No second scheduler. |
| BUILD | `cmake/package/CelleratorConfig.cmake.in`; operation-local `CMakeLists.txt`; this standalone host consumer | Granular `cmake/substrate` exports and installed consumer fixtures; preserve native component names where useful and explicitly repair any export change. |

The source-destination map is machine-readable in `source-destinations.json`.
`preservation.json` retains the 92-entry seed ledger exactly and adds every task
in the current parallel single-cell moonshot, numerical Baseplane moonshot and
ML2 plans. AMP permission lock, GEO closure, pending historical metadata and
frozen-interface owner reconciliation remain explicit obligations. Task scope
presence records are inventory evidence; they do not establish fresh numerical
acceptance. The declared `CE-MOON-NATIVE-READY` owner-receipt directory is absent at this
base; the map records that absence and retains its authority/history obligation.
Original sources/receipts stay in place until a destination supplies
the usable behavior and repaired consumers.

## Target dependency decision

Core host contracts/math do not depend on Baseplane. Existing concrete native
components remain `Cellerator::native_foundation`, `::indexed_mechanism`,
`::local_differential`, `::native_numeric` and `::product2` where available in the
configured native build. The source-only experimental `::moonshot` family
components are preserved until BUILD publishes equivalent supported consumers.
The host contract target is a source-tree preparation seam; it is not an
installed package or runtime guarantee.

`Baseplane::seq` remains independently buildable. The optional CE sequence
bridge links exact sequence and selected CE owners, without enabling Baseplane
representation. Baseplane representation/query links seq plus selected CE
components; GH experiments select their required CE components and optionally
BP. No package-wide reverse dependency or mandatory umbrella is introduced.
Shared registry/root build/install edits go to MERGE-A/BUILD integration owner.

## Focused gate

```sh
python3 -B planning/integrated-substrate-v1/adoption/check.py --build-dir /tmp/ce-is1-adopt-host
```

GCC 13.3 configured and built the actual host consumers; 3/3 CTest checks passed.
Strict warning, no-fast-math and no-FMA-contraction flags were used. The gate
also checks exact seed retention, completed-campaign inventory coverage and
source existence. CUDA execution, prepared residency, Torch, installed targets,
scientific qualification and performance were not run by ADOPT. Prior receipts
retain their original source/evidence limitations. Root separately binds and
runs the native activation/global-barrier gate, reconciles current authority,
and accepts/commits this patch before creating leaf worktrees from it.
