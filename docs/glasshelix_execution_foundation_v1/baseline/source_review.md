# NF1 source baseline

This is source and CMake-definition review at the commit pinned in
`source_observation.json`; it is not a new device or numerical qualification.
The JBC transition map and SS1/RU1 capability records are precedents whose live
source and target wiring were checked, not completion authority for NF1.

| Capability | Live owner and callable target | Classification and evidence limit |
| --- | --- | --- |
| Operation schema and preparation | `src/compute/operation/operation_core_v2/`; `Cellerator::operation_core` receives its source from `src/compute/CMakeLists.txt` | Implemented and source-linked in normal CUDA graph; no new run performed |
| Prepared programs | `src/execution/program/program_v2.cc`; `Cellerator::executable_program` | Existing owner; extend instead of a second program runtime |
| Value generations and planes | `src/execution/projection_value_plane/`; `Cellerator::projection_value_plane_v1` | Existing independent value/structure owner with source-linked validation/publication |
| SS1/RU1 calculus | `relation_semantics.cc`, `relation_calculus.cc`; `Cellerator::relation_semantics`, `Cellerator::relation_calculus` | Opt-in host-callable semantic calculus; not general nonlinear derivative execution |
| Prepared relation execution | `prepared_relation.cu`; `Cellerator::prepared_relation_cuda` | Real implementation, opt-in SS1/RU1 build; rejects widths outside 1/16 and storage outside f16 with f32 arithmetic |
| Edge gradient and update | prepared relation plus sm70 edge-value-gradient provider and relation-value readiness | Actual N16 bounded implementation, mathematical or explicitly half-rounded operands, mutable capture rejected |
| Broader NF1 API | planning contract header and demo | Developmental specification; not source-linked production implementation |

Root CMake is host-first: `CELLERATOR_ENABLE_CUDA=OFF` returns before the
ordinary CUDA runtime graph. SS1/RU1 are opt-in; RU1 enables testing and registers
host and GPU tests. README/AGENTS statements that CUDA is always required and
CTest is unused are older than this wiring and cannot determine NF1 gates.
Native NF1 registration is pending its assigned build owner. No empty CTest
invocation qualifies a capability. The default integrated demonstration must
still require actual CUDA as the NF1 mandate specifies.

The physical prepared-relation checks explicitly reject duplicate endpoint
edges and unsupported numeric semantics. N32/N64 provider source elsewhere
does not widen this prepared-pair API. A source file or target name alone does
not establish external consumption, arbitrary-width FP32, nonlinear n-ary
execution, JVP, second-direction actions, or mutable graph capture.

Keep CSG1 semantic identity and exact covers, operation core v2, the execution
session, program v2, and projection-value generation owners. Preserve legacy
candidates as measured alternatives and RU1 as a bounded regression witness.
Do not activate CE-AMP, rewrite historical worktrees, or use a compiler shell
as evidence of executed numerical semantics.
