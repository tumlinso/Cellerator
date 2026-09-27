# Source ledger

This package separates **observed source**, **inference from that source**, and **proposed work**. Repository reads were made through Project Control on 27 September 2026. File ranges below identify the inspected material; they are not claims that all repository implementations or runtime paths were tested.

## Live source identities

| Repository | Observed commit | Todo revision | Working tree |
|---|---|---:|---|
| GlassHelix | `a7d330b46b76128b3c8dba069a3d8367d0d1258f` | 237 | Clean |
| Cellerator (including CelleraTorch) | `9a850576f1e46f15c5095011b34571a41e4f8d7a` | 7417 | Clean |
| project-control | `366e85e88d1585e511c12bb856f7fd64e77ff3d5` | 803 | Dirty; working-tree reads |
| skills | `fc9d083afd614b43f16bdd1557243015a6ecc899` | 887 | Clean |

Refresh relevant identities at adoption. These are research provenance, not a demand to roll back current work to old commits. The observer reported `stale_legacy_state_filtered`; this is not authorization to repair or retire historical work.

## Source anchors

**S1 — Cellerator scope.** `cellerator:scope.md`, lines 1–158. Native biological domains, axes, relations, geometry, values, operators, planning, training/optimizer primitives and precision; CT adaptation; CellShard/Baseplane boundaries. Explicitly excludes a general-purpose tensor framework. This is an ownership document, not proof every scoped capability is fully implemented.

**S2 — Actual CelleraTorch training boundary.** `cellerator:components/CelleraTorch/docs/autograd_ops.md`, lines 1–49; `components/CelleraTorch/src/autograd_ops.cu`, lines 123–253, especially 207–253. The documented combined N16 training route performs native parameter updates during backward, returns only dense-input gradients and excludes a second Torch optimizer over the same native parameters. Source calls `run_training_program` and returns undefined gradients for the other arguments. This is the decisive reason a new composable route is needed.

**S3 — CelleraTorch/parameter integration.** `components/CelleraTorch/README.md`, lines 1–85; `include/Cellerator/parameters.hh`, lines 1–74; `components/CelleraTorch/CMakeLists.txt`, lines 35–160. Framework wrappers/views consume CE-owned native buffers, identities and readiness. Native descriptors distinguish storage shape/stride/type/role from structure and generation. Existing libtorch targets are not evidence of a complete convenient Python model API.

**S4 — NF1A integrated GH foundation.** `glasshelix:docs/nf1_adaptive/qualification/INTEGRATED_TOOLBOX.md`, lines 1–46; `CMakeLists.txt`, lines 1–158; Project Control's current GH-NF1A run record. Supplied-system assembly, actual RK4/direct-map evolution, observations, selected responses, joint candidates and source-bound replay; run recorded completed. Python is a bounded host-JVP surface. Acceptance records describe earlier executed tests, not tests rerun for this bootstrap.

**S5 — Reconfirmed GH repairs.** `include/GlassHelix/interrogation/diagnostics.hh`, lines 1–15; `include/GlassHelix/inference/inference.hh`, lines 74–134. The SVD path takes contiguous chunks of column-major `VT` as right null vectors. Assimilation persists ordinary weights rather than authoritative log weights. These source patterns support the proposed regression repairs; this preparation did not execute the repaired native code.

**S6 — GH response composition.** `include/GlassHelix/dynamics/response_plan.hh`, lines 1–195. Saved primal owners, structure/candidate IDs and generation checks; declared direct-map JVP/VJP and RK4 response composition. Preserve scientific identity and the actual forward being differentiated. This is not a general learned-parameter training frontend.

**S7 — Native prepared relations.** `cellerator:include/Cellerator/compute/operation/prepared_relation.hh`, lines 1–113. Typed axes, logical edges, prepared forward/transpose pair, instance creation over shared topology, f32/f16 bindings, epochs, values and explicit readiness/reuse. Test the selected native implementation; a declaration alone does not establish end-to-end training support.

**S8 — Native derivatives/updates.** `include/Cellerator/compute/operation/relation_update.hh`, lines 1–92; `include/Cellerator/compute/operation/differential/local_arithmetic.hh`, lines 1–83. Edge-gradient and mutation contracts, gradient stamps and leases; local JVP/VJP with explicit contribution assembly. The selected new model needs its whole derivative path qualified rather than inferred from these primitive interfaces.

**S9 — Typed multi-input mechanisms.** `include/Cellerator/compute/operation/indexed_mechanism/incidence.hh`, lines 1–48; `evaluators.hh` in the same directory, lines 1–89. Ordered/repeated argument slots, explicit output effects, executable sum/product/first-minus-product-tail vocabulary, host/CUDA evaluation and prepared stages. This inspected indexed API does not itself expose the complete learned-parameter gradient path proposed for Candidate B.

**S10 — Actual compiler/native bridge.** `docs/relation_update_spine_v1/compiler_boundary.md`, lines 1–118. Bounded independently parsed relation/update closure lowers to the common native calculus and effect ordering. The document explicitly limits this bridge; it is not full `.cell` compilation or arbitrary Torch graph compilation.

**S11 — Language/compiler direction.** `docs/language/cellerator-language-specification.md`, lines 43–133; `planning/cellerator-compiler-preledger-v1/README.md`, lines 1–76. The language is a proposed explicit biological C++ extension; semantic facts survive lowering and representative profiles propose rather than certify correctness. The compiler, driver, IR, APIs and standard library are CE responsibilities. The historical large compiler package is reference context, not work to reactivate here.

**S12 — Current bootstrap/workflow contracts.** `project-control:docs/ADAPTIVE_EPICS.md`, lines 1–65; `docs/CODEX_SETUP.md`, lines 58–169; `src/project_control/cli.py`, lines 120–125 and 270–305; `src/project_control/services/planning.py`, lines 165–266. `skills:todo-orchestrator/references/workflow-v3-operations.md`, lines 1–70; `todo-orchestrator/todo_orchestrator/workflow/roles.py`, lines 1–102; `todo-orchestrator/todo_orchestrator/plan.py`, lines 1–200. Coarse outcomes, schema v3 runs/lanes, implementer role constraints, gate binding, exact plan validation/apply front door. Local plan validation remains required at adoption.

## External primary references

These references inform integration and the evidence audit, not claims of product implementation.

**W1.** PyTorch, *Custom C++ and CUDA Operators*: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html — registered external operators, autograd registration and separate registration/gradient testing.

**W2.** PyTorch, *Custom Operators*: https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html — reuse built-in composition where adequate; integrate external operators deliberately.

**W3.** PyTorch, `torch.library` reference: https://docs.pytorch.org/docs/stable/library.html — operator schemas, mutation/aliasing, autograd and optional framework-transform registration. Use the locally qualified Torch version; these current pages do not authorize upgrading the host.

**W4.** *Single-cell lineage tracing to uncover gene-regulatory networks driving cell fate specification*, Nature Biotechnology, CellTag-multi: https://www.nature.com/articles/s41587-023-01931-4 — RNA/chromatin lineage-linked reprogramming experiment and methods. Protocol/data facts are kept in `EXPERIMENT.md`; the ML constructions in this package are proposals, not findings of that study.

**W5.** GEO GSE216518, iEP RNA: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE216518 — sample identities and downloadable processed data. Includes distinct day-14 perturbation samples; do not pool them silently into the common-forcing experiment.

**W6.** GEO GSE217119, iEP ATAC: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE217119 — separate ATAC samples and processed data. No RNA-to-ATAC same-cell join is implied by sharing a time label.

The relevant GEO records were read; data archives and clone tables were not downloaded, joined or fitted during this preparation. Live native/GPU and biological validation are implementation outcomes, not properties of this archive.
