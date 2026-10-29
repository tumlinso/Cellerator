"""CPU reference: cell state -> selected experts -> placement -> original owners.

Run: PYTHONPATH=python python examples/framework/scheduled_experts.py
The resource labels describe CPU callbacks, not physical GPU measurements.
"""
from cellerator import Cell, Model
from cellerator.execution import (
    BoundExecution, DeviceSpec, ExecutionEngine, ExpertPlacement, ExpertSpec,
    ExpertWorklist, HardwareTopology, LinkSpec, ParameterKey, ParameterRevision,
    ParameterStore, PlacementPlanner, ResidencyManager, SelectedWork,
    reduce_execution_outputs,
)


def main():
    receiving_model = Model(lambda state, context: context.inputs)
    cells = {name: Cell(receiving_model, value, id=name)
             for name, value in (("cell-a", 2.0), ("cell-b", 5.0))}
    slow, fast = ParameterKey("slow", 0), ParameterKey("fast", 0)
    residency = ResidencyManager({"cpu-worker": 1024}, lambda payload, *_: payload)
    store = ParameterStore(residency)
    store.register(ParameterRevision(slow, 0.5, 8))
    store.register(ParameterRevision(fast, 2.0, 8))

    # The model/caller selects these sparse occurrences, including overlapping
    # expert contributions. The placement planner cannot change this choice.
    work = ExpertWorklist.from_selections([
        SelectedWork(fast, "cpu-worker", "cell-b", 1.0),
        SelectedWork(slow, "cpu-worker", "cell-a", 0.25),
        SelectedWork(fast, "cpu-worker", "cell-a", 0.75),
    ])
    topology = HardwareTopology([
        DeviceSpec("host", kind="host", memory_bytes=1024),
        DeviceSpec("cpu-worker", memory_bytes=1024),
    ], [LinkSpec("host", "cpu-worker", 1e9, bidirectional=True)])
    planner = PlacementPlanner(topology, [ExpertSpec(slow, 8), ExpertSpec(fast, 8)],
        [ExpertPlacement(slow, "host", executable=False),
         ExpertPlacement(fast, "host", executable=False)])
    plan = planner.plan(work.demands(activation_bytes_per_row=8, output_bytes_per_row=8))
    bound = BoundExecution(plan, work, work_revision="cell-membership-1")

    def numerical_backend(request):
        return tuple(request.inputs[selected.row_id] * request.parameters
                     for selected in request.selections)

    inputs = {key: cell.state for key, cell in cells.items()}
    with ExecutionEngine(store, numerical_backend) as engine:
        result = engine.submit(bound, inputs, work_revision="cell-membership-1").result()
    # Explicit weighted reduction restores each result to its biological owner.
    updates = reduce_execution_outputs(result, owner=lambda selected: selected.row_id)
    for key, value in updates.items():
        cells[key].advance(dt=1.0, inputs=value)
    assert cells["cell-a"].state == 3.25
    assert cells["cell-b"].state == 10.0
    # Publish trainable state separately; an existing bound plan remains revision 0.
    store.update(fast, lambda old: ParameterRevision(
        ParameterKey(old.key.expert_id, 1), 1.9, 8,
        master_parameters=1.9, gradients=1.0, optimizer_state={"step": 1}))
    print({key: cell.state for key, cell in cells.items()})
    print("Current fast revision:", store.current("fast").revision)


if __name__ == "__main__":
    main()
