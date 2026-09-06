# Lanes, dependencies and integration

The schema-3 native run declares eight first-class lanes: coordinator, core contracts, native pair, readiness, WMMA/contraction, frontend, independent validation and integration. These are ownership/coordination roles, not eight background jobs started by this package. Local worker children never stand in for first-class lanes.

The foundation proceeds I01 -> C01 -> C02 -> C03 -> I02. After the foundation is integrated, NATIVE, READINESS, WMMA, FRONTEND and VERIFY can begin independently. All changes to `prepared_relation.cu` remain within NATIVE, while readiness has separate runtime components and WMMA separate providers. Central build files remain INTEGRATE-owned. C02 owns public native headers initially; any necessary later interface change is a coordinated serial handoff, not concurrent editing by dependent lanes.

Within each lane the task list is a serial queue. Those queue edges are part of offline DAG validation alongside explicit task dependencies. The root epic is not a prerequisite of its own leaves. Barriers are derived from actual task checkpoints and do not pre-reach themselves. Interface ownership is internal development coordination, not a promise of public ABI stability.

The fan-in is I03 after N06, R03, W05, F04 and V05. Test authoring can happen before final build integration; final hardware acceptance cannot. I04 finishes the example against real core, I05 executes device/sanitizer acceptance, I06 records lifetime performance, I07 removes superseded ownership, and I08 reruns final gates and closes. No gate may use a historical receipt as current execution evidence.

Use the existing project GPU lease for every accelerator run. The package's logical evidence lock only serializes its own acceptance; it does not allocate a GPU or exclude other workloads. No new resource-class inventory is invented. Existing JBC/CCP worktrees, old runs, pending patches and Ampere permission gates are untouched. Baseline SS1 completion is source evidence, not a cross-authority dependency to be manufactured in a new plan.

The validator rejects overlapping unordered write paths, duplicate task IDs, missing/duplicate lane assignments, queue-induced cycles, missing barrier references and inconsistent machine projections. Exact human task sheets supply additional acceptance detail; native task records point to them and cannot be applied meaningfully without the package installed at its recorded path.
