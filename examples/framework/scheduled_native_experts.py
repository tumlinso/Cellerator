"""CPU execution: Python state/plan/residency -> native dispatch -> original owners.

Requires the optional built cellerator._conditional_native extension.
Run: PYTHONPATH=python python examples/framework/scheduled_native_experts.py
"""
import numpy as np
from cellerator import DynamicalCell, DynamicalModel
from cellerator.adapters.conditional import NativeConditionalBackend, NativeIdentityMap
from cellerator.execution import (
    BoundExecution, DeviceSpec, ExecutionEngine, ExpertPlacement, ExpertSpec,
    ExpertWorklist, HardwareTopology, LinkSpec, ParameterKey, ParameterRevision,
    ParameterStore, PlacementPlanner, ResidencyManager, SelectedWork,
)


def run(native_module=None):
    model = DynamicalModel(lambda state, context: context.inputs)
    cells = {name: DynamicalCell(model, np.array([value], dtype=np.float16), id=name)
             for name, value in (("cell-a", 2), ("cell-b", 5))}
    old, other = ParameterKey("gain", "revision-a"), ParameterKey("other", "revision-a")
    identities = NativeIdentityMap(
        {old: (100, 7), ParameterKey("gain", "revision-b"): (100, 8), other: (2, 7)},
        {("cpu", "cell-a"): (0, 10, 99), ("cpu", "cell-b"): (0, 20, 7)})
    residency = ResidencyManager({"cpu": 1024}, lambda parameters, *_: parameters)
    store = ParameterStore(residency)
    store.register(ParameterRevision(old, np.float32(3), 4))
    store.register(ParameterRevision(other, np.float32(2), 4))
    work = ExpertWorklist.from_selections([
        SelectedWork(old, "cpu", "cell-b", -0.5),
        SelectedWork(other, "cpu", "cell-a", 1.0),
        SelectedWork(old, "cpu", "cell-b", 1.5),
    ])
    topology = HardwareTopology([
        DeviceSpec("host", kind="host", memory_bytes=1024),
        DeviceSpec("cpu", memory_bytes=1024),
    ], [LinkSpec("host", "cpu", 1e9, bidirectional=True)])
    plan = PlacementPlanner(topology, [ExpertSpec(old, 4), ExpertSpec(other, 4)],
        [ExpertPlacement(key, "host", executable=False) for key in (old, other)]).plan(
            work.demands(activation_bytes_per_row=2, output_bytes_per_row=4))
    # A newer canonical version cannot silently change an already selected key.
    store.publish(old, ParameterRevision(ParameterKey("gain", "revision-b"), np.float32(9), 4))
    visited = []

    def numerical(key, parameters, values, selection_indices):
        assert values.dtype == np.float32  # explicit native FP16-storage policy
        visited.extend(selection_indices)
        assert key in (old, other)
        return np.ascontiguousarray(values * parameters)

    backend = NativeConditionalBackend(identities, numerical, input_features=1,
        output_features=1, native_module=native_module)
    with ExecutionEngine(store, backend) as engine:
        result = engine.submit(BoundExecution(plan, work, "membership-1"),
            {("cpu", name): cell.state for name, cell in cells.items()},
            work_revision="membership-1").result()
    np.testing.assert_array_equal(np.stack(result.outputs), [[15], [4], [15]])
    reduced = {}
    for selection, output in zip(result.selections, result.outputs):
        reduced[selection.row_id] = reduced.get(selection.row_id, 0) + selection.weight * output
    for name, value in reduced.items():
        cells[name].advance(inputs=value)
    assert sorted(visited) == [0, 1, 2]
    assert cells["cell-a"].state.tolist() == [4]
    assert cells["cell-b"].state.tolist() == [15]
    return {name: cell.state.tolist() for name, cell in cells.items()}


if __name__ == "__main__":
    print(run())
