# Actor-local state and footprint placement prototype

CE-MOON-STATE supplies a CPU reference and C++17 host view. It introduces no
production owner, native adapter, CUDA execution, optimizer update or timing
claim. Native integration must translate Cellerator stable identities and
generation APIs at its owner boundary; these experimental identities are not ABI.

`state_reference.py` provides:

- `Coordinate(actor, local_slot, incarnation)` and separate structure, value,
  activity and parameter generations in `Epoch`.
- `StateOwner(values, coordinates, epoch)`, borrowing a floating rank-1 ndarray,
  and `StateView(owner, slots)`, a physical ordering of `Slot` bindings. A slot
  carries independent support, capacity, optional activity and residual placement.
  Physical replicas retain the same identity and canonical offset. Different
  actors or incarnations cannot become aliases.
- `gather()` reads a saved generation; `pullback()` sums replica cotangents to
  their canonical owner. `assemble(contributions, output_coordinates)` creates a
  fresh output from exactly one occurrence of each supplied contribution ID.
  Multiple distinct contributions may reduce into a declared supported output.
- `Footprint` and `pack(footprints, width=16)` use declared tagged read/write/context
  support with a traffic/padding/reduction proxy. Ordering and tie breaks are
  deterministic under input permutation. Each logical identity appears once;
  grouping changes placement, without tying weights or interpreting biology.
  This direct all-pairs shortlist is for small prototypes, not a scalable planner.
- `local_port_transport(states, encoders, decoders, adjacency, local_fields=None)`
  accepts lists of actor-private widths, independent `E_i[P,H_i]` and `D_i[H_i,P]`,
  and supplied `A[dst,src]`. It computes `phi_i + D_i sum_j A_ij E_j h_j`. The
  optional fields are already evaluated local law values. Their own derivatives
  and conjugation under basis changes belong to the caller.

`state_view.hh` provides borrowed C++ pointer/count maps, validation against the
live owner epoch/coordinate array, gather and replica pullback. The caller owns
storage lifetimes and synchronization. Python similarly keeps references alive
without copying canonical values. External writes must increment the proper
owner generation via `touch`; direct ndarray writes cannot be intercepted. The
saved epoch check is admission validation, not a concurrent reader lock.

The view borrows current storage and rejects reuse after a generation change.
Both Python `gather()` and C++ `gather()` return independent value copies. A
caller saving primal operands must retain those copies together with the epoch
and actual stored precision; holding a view does not preserve old values. This
prototype checks that a gathered copy survives later owner mutation. It does
not implement a framework tape or concurrent snapshot protocol.

Run the combined CPU check from the repository root:

```sh
CUDA_VISIBLE_DEVICES='' /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B experiments/moonshot-parallel-v1/state/check_cpu.py
```

The check runs 11 Python tests, then compiles and runs the C++17 host witness in
a temporary directory. Torch is used only as an independent CPU oracle for the
port algebra and gradients. Tests cover zero-but-supported inactive coordinates,
reserved capacity, stale epochs/incarnations, invalid aliasing, output ownership,
replica adjoint identity, permutation-stable packing, private basis changes,
heterogeneous widths and distinct encoder parameter gradients. The receipt and
log describe direct execution; a required Project Control gate must still be
bound and executed before task completion.
