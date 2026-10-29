"""Focused optional CPU dispatch-binding check, independent of Torch/CUDA."""
import importlib
import numpy as np

def main():
    try:
        native = importlib.import_module("_conditional_native")
    except ModuleNotFoundError:
        native = importlib.import_module("cellerator._conditional_native")
    records = [
        (20, 1, 0, 10, 1, 0.5),
        (10, 2, 0, 10, 1, -1.0),
        (20, 1, 1, 20, 2, 2.0),
        (10, 3, 1, 20, 2, 0.0),
        (20, 1, 0, 10, 1, 0.25),
    ]
    plan = native.DispatchPlan(records)
    assert len(plan) == 5 and plan.origin_count == 2
    assert [(key.expert_id, key.revision) for key in plan.experts] == [(10, 2), (10, 3), (20, 1)]
    assert plan.expert_offsets == [0, 1, 2, 5]
    assert plan.dispatch_to_selection == [1, 3, 0, 2, 4]
    assert plan.dispatch_origin_slots == [0, 1, 0, 1, 0]
    assert plan.dispatch_weights == [-1.0, 0.0, 0.5, 2.0, 0.25]
    for dispatch, selection in enumerate(plan.dispatch_to_selection):
        assert plan.selection_to_dispatch[selection] == dispatch

    for dtype in (np.float16, np.float32, np.float64):
        inputs = np.array([[1, 3], [2, 4]], dtype=dtype)
        working_dtype = np.float32 if dtype == np.float16 else dtype
        dispatched = native.gather_inputs(plan, inputs)
        assert dispatched.dtype == working_dtype
        np.testing.assert_array_equal(dispatched, inputs[plan.dispatch_origin_slots])

        visited = []

        def expert(key, values, origins, selection_indices):
            assert values.dtype == working_dtype and values.flags.c_contiguous
            for row, selection in enumerate(selection_indices):
                visited.append(selection)
                record = records[selection]
                assert (key.expert_id, key.revision) == record[:2]
                assert (origins[row].rank, origins[row].instance, origins[row].row) == record[2:5]
            scale = key.expert_id + key.revision / 4
            return (values[:, :1] * scale + values[:, 1:2]).astype(working_dtype)

        result = native.execute(plan, inputs, 1, expert)
        assert result.shape == (2, 1) and result.dtype == working_dtype
        assert sorted(visited) == list(range(len(records)))
        expected = np.zeros((2, 1), dtype=working_dtype)
        per_dispatch = np.empty((len(records), 1), dtype=working_dtype)
        for selection, record in enumerate(records):
            expert_id, revision, rank, instance, row, weight = record
            origin = 0 if (rank, instance, row) == (0, 10, 1) else 1
            value = inputs[origin, 0] * working_dtype(expert_id + revision / 4) + inputs[origin, 1]
            expected[origin, 0] += working_dtype(weight) * value
            per_dispatch[plan.selection_to_dispatch[selection], 0] = value
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(native.combine(plan, per_dispatch), expected)
        occurrences = native.execute_occurrences(plan, inputs, 1, expert)
        assert occurrences.shape == (len(records), 1) and occurrences.dtype == working_dtype
        np.testing.assert_array_equal(occurrences, per_dispatch[plan.selection_to_dispatch])

        gradients = np.array([[2], [-3]], dtype=working_dtype)
        transposed = native.transpose_combine(plan, gradients)
        lhs = float(np.sum(expected * gradients))
        rhs = float(np.sum(per_dispatch * transposed))
        np.testing.assert_allclose(lhs, rhs)

    empty = native.DispatchPlan([])
    assert empty.expert_offsets == [0] and empty.origin_offsets == [0]
    assert native.execute(empty, np.empty((0, 2), np.float32), 1, lambda *args: None).shape == (0, 1)
    assert native.execute_occurrences(empty, np.empty((0, 2), np.float32), 1, lambda *args: None).shape == (0, 1)

    for records_with_bad_weight in (
        [(1, 0, 0, 0, 0, float("nan"))], [(1, 0, 0, 0, 0, float("inf"))]
    ):
        try:
            native.DispatchPlan(records_with_bad_weight)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite route weight accepted")

    try:
        native.execute(plan, np.ones((2, 2), np.float32), 1,
                       lambda key, values, origins, selections: np.ones((len(values), 1), np.int32))
    except TypeError:
        pass
    else:
        raise AssertionError("callback output dtype contract was not enforced")


if __name__ == "__main__":
    main()
