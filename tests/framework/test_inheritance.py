import unittest

import numpy as np

from cellerator.experimental.inheritance import (
    BinomialPartition,
    CopyInheritance,
    DivisionContext,
    FieldwiseInheritance,
    FractionPartition,
    ResetField,
    divide_cell,
    divide_population,
    paired_fork_callbacks,
)
from cellerator.execution import Population
from cellerator.modeling import DynamicalCell, DynamicalModel


class FixedRng:
    def __init__(self, count):
        self.count = count
        self.calls = []

    def binomial(self, n, p):
        self.calls.append((n, p))
        return self.count


class InheritanceTests(unittest.TestCase):
    def test_count_partition_conserves_and_uses_supplied_rng(self):
        rng = FixedRng(3)
        left, right = BinomialPartition(0.7).divide(11, DivisionContext(rng=rng))
        self.assertEqual((left, right), (3, 8))
        self.assertEqual(rng.calls, [(11, 0.7)])

    def test_sparse_model_arrays_keep_their_container_and_conserve(self):
        rng = np.random.default_rng(11)
        counts = np.array([0, 3, 17, 200], dtype=np.int64)
        first, second = BinomialPartition(0.35).divide(counts, DivisionContext(rng=rng))
        left, right = FractionPartition(0.7).divide(np.array([2.0, 7.0]), DivisionContext())
        np.testing.assert_array_equal(first + second, counts)
        np.testing.assert_allclose(left + right, [2.0, 7.0])
        self.assertEqual(first.dtype, counts.dtype)

    def test_fieldwise_rules_support_asymmetry_reset_and_copy(self):
        rule = FieldwiseInheritance({
            "mass": FractionPartition(0.25),
            "accessible": ResetField(lambda value, context, daughter: daughter == 0),
            "memory": CopyInheritance(),
        })
        left, right = rule.divide(
            {"mass": 8.0, "accessible": "old", "memory": [1, 4]},
            DivisionContext(daughter_ids=("d0", "d1")),
        )
        self.assertEqual((left["mass"], right["mass"]), (2.0, 6.0))
        self.assertEqual((left["accessible"], right["accessible"]), (True, False))
        self.assertEqual(left["memory"], [1, 4])
        self.assertIsNot(left["memory"], right["memory"])

    def test_mutable_state_is_not_aliased_between_daughters(self):
        parent = DynamicalCell(DynamicalModel(lambda state, context: state), {"history": [1], "memory": {"x": 2}}, id="p")
        left, right = divide_cell(parent, CopyInheritance(), daughter_ids=("l", "r"))
        left.state["history"].append(2)
        left.state["memory"]["x"] = 7
        self.assertEqual(right.state, {"history": [1], "memory": {"x": 2}})
        self.assertEqual(parent.state, {"history": [1], "memory": {"x": 2}})
        self.assertIs(left.model, right.model)

    def test_population_division_performs_one_conserving_draw(self):
        model = DynamicalModel(lambda state, context: state, parameters=object())
        population = Population([DynamicalCell(model, {"count": 10, "x": [1]}, id="p")])
        rng = FixedRng(4)
        rule = FieldwiseInheritance({"count": BinomialPartition()})
        first, second = divide_population(
            population, "p", rule, daughter_ids=("a", "b"), rng=rng,
        )
        self.assertEqual((first.state["count"], second.state["count"]), (4, 6))
        self.assertEqual(population.ids, ("p", "a", "b"))
        self.assertIs(first.model.parameters, model.parameters)
        self.assertEqual(rng.calls, [(10, 0.5)])

    def test_fork_callbacks_share_one_paired_random_draw(self):
        model = DynamicalModel(lambda state, context: state)
        parent = DynamicalCell(model, {"count": 10}, id="p")
        population = Population([parent])
        rng = FixedRng(2)
        first, second = paired_fork_callbacks(
            FieldwiseInheritance({"count": BinomialPartition()}), rng=rng,
        )
        daughter_a = population.fork("p", inherit=first, id="a")
        daughter_b = population.fork("p", inherit=second, id="b")
        self.assertEqual((daughter_a.state["count"], daughter_b.state["count"]), (2, 8))
        self.assertEqual(rng.calls, [(10, 0.5)])

    def test_integer_fraction_partition_is_rejected(self):
        with self.assertRaises(TypeError):
            FractionPartition().divide(8, DivisionContext())


if __name__ == "__main__":
    unittest.main()
