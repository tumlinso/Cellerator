"""Focused native extension check; executable standalone or through CTest."""
from __future__ import annotations

import importlib
import numpy as np

def main() -> None:
    try:
        native = importlib.import_module("_modeling_native")
    except ModuleNotFoundError:
        native = importlib.import_module("cellerator._modeling_native")
    relation = native.SparseRelation([
        (20, [(0, 100), (0, 200), (1, 300)]),
        (10, [(0, 200), (0, 200), (1, 100)]),
    ], relation_id=7, structure_id=9, structure_epoch=4)
    assert relation.edge_ids == [10, 20]
    assert relation.query([(0, 200)]) == [10, 20]
    assert relation.query([(0, 100), (1, 300)], require_all=True) == [20]
    active = native.ActivityState(relation)
    worklist = active.worklist()
    active.set_active([10], False)
    assert active.generation == 1 and active.worklist().edge_ids == [20]
    assert worklist.edge_ids == [10, 20]
    layout = native.EntityLayout([300, 100, 200])
    assert layout.permutation_to(native.EntityLayout([100, 200, 300])) == [1, 2, 0]
    plan = native.IncidencePlan(worklist, layout)
    assert plan.offsets == [0, 3, 6]
    for dtype in (np.float32, np.float64):
        values = np.array([[3, 30], [1, 10], [2, 20]], dtype=dtype)
        np.testing.assert_array_equal(native.gather(plan, values),
                                      [[2, 20], [2, 20], [1, 10], [1, 10], [2, 20], [3, 30]])
        reduced = native.joint_reduce(plan, values, role=0)
        assert reduced.dtype == dtype
        np.testing.assert_array_equal(reduced, [[4, 40], [3, 30]])
        np.testing.assert_array_equal(native.joint_reduce(plan, values, reduction="mean", role=0),
                                      [[2, 20], [1.5, 15]])
        np.testing.assert_array_equal(native.scatter_add(plan, reduced, role=1),
                                      [[3, 30], [4, 40], [0, 0]])
        np.testing.assert_array_equal(native.joint_reduce_scatter(plan, values, input_role=0, output_role=1),
                                      [[3, 30], [4, 40], [0, 0]])
        flat = np.array([3, 1, 2], dtype=dtype)
        assert native.joint_reduce(plan, flat).shape == (2,)

    field = native.Float32Field(layout, np.array([[3, 30], [1, 10], [2, 20]], dtype=np.float32))
    field.update([200], np.array([[8, 80]], dtype=np.float32))
    assert field.generation == 1
    np.testing.assert_array_equal(field.values, [[3, 30], [1, 10], [8, 80]])
    field.values[0, 0] = 99
    assert field.values[0, 0] == 3

    dynamic = native.DynamicRelation(relation)
    dynamic.stage_add(30, [(0, 300), (1, 100)])
    dynamic.stage_remove(10)
    replacement = dynamic.publish()
    assert replacement.edge_ids == [20, 30] and replacement.structure_epoch == 5
    assert dynamic.snapshot.edge_ids == [20, 30]
    active.set_active([20], False)
    rebased = active.rebase(replacement)
    assert rebased.worklist().edge_ids == [30]
    assert active.rebase(replacement, new_edges_active=False).worklist().edge_ids == []
    assert plan.edge_ids == [10, 20] and plan.structure_epoch == 4
    try:
        dynamic.stage_add(10, [(0, 300)])
    except ValueError:
        pass
    else:
        raise AssertionError("retired native IDs must not be reused")

    for invalid in (np.ones((3, 2), dtype=np.int32), np.ones((3, 4), dtype=np.float32)[:, ::2]):
        try:
            native.gather(plan, invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("dtype and contiguous layout contracts must be enforced")


if __name__ == "__main__":
    main()
