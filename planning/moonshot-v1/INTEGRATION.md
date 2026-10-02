# Integration: stage, adapt, then publish a native capability

## Do not install an alternative runtime

The package's `payload/cellerator` tree is a standalone experimental workspace to make copying and compilation easy. Its `moonshot` namespace, coordinate IDs and raw views are **not** a second public identity or memory system. During native integration, adapt them to the actual CE stable IDs, memory views, lifecycle contracts, parameter owners and prepared program. Keep old proven operations intact.

The installer copies the full package under a new `planning/moonshot-v1` directory in each repository. The claimed implementing task can then use:

```sh
# In the Cellerator repository, after taking the matching claim:
python3 planning/moonshot-v1/bootstrap/stage_source.py --role cellerator --root . --apply

# In the GlassHelix repository, after taking its own claim:
python3 planning/moonshot-v1/bootstrap/stage_source.py --role glasshelix --root . --apply
```

Default without `--apply` is a preview. These commands copy only to a **new** `experiments/moonshot-v1`. They never overwrite the core or old experiment directories. Source integration remains agent work under current scopes.

## Native mapping

| Prototype element | Existing CE home / seam | Integration decision |
|---|---|---|
| Nonowning state packet | `memory/view.hh`, `execution/atom_plane/mutable_state_plane_v1.hh`, semantic state/axis types | Use canonical IDs and value/structure generations; explicit mapping from actor-local slots to physical offsets |
| Matrix patch / micro-MMA | `compute/architecture/providers/nvidia/sm70/`, operation/candidate catalog | New explicit experimental candidate, not a silent replacement of all matmul |
| Process packet | `compute/operation/indexed_mechanism/` | Reuse incidence, coefficient owner and assembly; preserve argument multiplicity |
| Footprint placement | `geometry/optimizer.hh` and related discovery/planning | Adapt candidate generation and cost evidence; do not call sample-detection similarity a dependency graph |
| New operation contract | `compute/operation/native_foundation_contract.hh` | Bind roles, axes, precision, effects and supported derivative capabilities |
| Prepared execution | `execution/program/program_v2.h` | Add typed admission/launch payloads over existing prepared stages |
| Learning adapter | `components/CelleraTorch/src/mechanism.cc` patterns | Register an actual callable op, saved tensor/tape and native-parameter gradient path |
| Promotion publication | Existing parameter/state owner, generations, readers, prepared structure | Stage all changes; publish only after readers/tapes make it legal; do not mutate native owner through raw Torch data writes |
| GH scientific model | `src/system/assembly.cc`, current Python/Torch route | Keep scientific roles/evidence outside numerical kernel implementation |

No particular new public filename is imposed. The implementing owner should choose names consistent with the live tree rather than preserve a package-local demo name for its own sake.

## Actual delivered code

| Payload | Status / what is still needed |
|---|---|
| `reference/moonref/operators.py` | Executed NumPy math references for patch derivatives, private ports, products, raw-MMA coordinate map, feedback solve, polynomial field and linear flow |
| `reference/moonref/promotion.py` | Executed supplied linear refactor/quotient, adjoint replica reduction, epoch guard and residual regrowth witnesses; not a nonlinear discovery engine |
| `reference/moonref/packing.py` | Executed illustrative CPU footprint shortlist/placement; not native optimizer compatibility or measured cost |
| `reference/moonref/delta.py` | Executed fixed-map last-transmitted ledger with instantaneous discrepancy bound |
| `include/moonshot/state_packet.hh` | Host C++ layout compiled; replace demo identities/views with actual CE types at integration |
| `kernels/quad_mma_sm70.cu` | Substantial inline-PTX source, uncompiled/unrun; device smoke provided |
| `kernels/patch16_sm70.cu` | Two-WMMA nonlinear patch source, uncompiled/unrun; needs same-stored-half reference and derivative integration |
| `kernels/product2_jvp_sm70.cu` | Two-input product/JVP with warp-local shared gathers, uncompiled/unrun; needs native validated input extents and assembly |
| GH `actor_fields.py` | Executed ordinary Torch prototypes: plain patches, private-port actors, live residual branch and linear-flow model |

## Admission before launch

The standalone CUDA functions are seeds with basic argument checks, not full public admission contracts. The CE adapter must validate dtype, element extents, device, contiguous/strided layout as required, alignment, index bounds, output aliasing and all capacities **before** launching. A process-index pointer alone does not establish that it can read the supplied state. No per-edge allocator or callback is introduced.

For full-tile kernels, padding is initialized deterministically. Predicated-excluded entries must not read invalid state or propagate NaNs from deliberately absent arguments. State and parameter generations are bound to the recorded computation.

## Derivatives and precision

The default numerical reference is float64. The patch CUDA seed uses half operands and a half nonlinear intermediate with FP32 accumulation/output. Add an explicit stored-half/STE reference rather than pretending those functions are bit-identical. Preserve the existing CE-ML2-TRAIN distinction between derivatives of stored values and the chosen quantization surrogate.

A native forward primitive is not a native backward primitive. Publish a table of actually implemented forward/input-VJP/parameter-VJP/JVP/second-order support. Use the formulas in MECHANISMS.md and recorded input/parameter values, not live parameters after an optimizer update.

## Regrowth and masks

An initially zero outgoing map is **still structurally trainable**. Its forward contribution may be zero while its parameter gradient is not. Keep trainable slots in derivative support. If forward work is omitted, the backward path must retain or reconstruct the hidden feature needed for the outgoing gradient. Do not recycle a slot by merely setting its floating-point value to zero.

Physical replicas read one canonical generation and sum their pullbacks to the canonical owner. A model refactoring that creates genuinely distinct child actors is different: it publishes distinct logical IDs and a declared state map.

## Prototype receipts, not qualification campaigns

`machine/capability-receipt.template.json` is an exchange format for these tasks, not a new native authority. Fill only actual implementation facts: source revisions, op name, model class, shapes, precision, callable path, derivatives, tiny checks, timing status and unsupported cases. GH consumes that receipt before claiming a native path. It is acceptable to keep a correct reference or explicit experimental selector when performance is not yet known.

The existing planner's stable choices should not change on theoretical throughput alone. New prototypes can be selected explicitly for experimentation, and different model families cannot be swapped by a performance planner as though they were equivalent kernels.
