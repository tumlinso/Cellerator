"""CPU mocks for SparseSupportModule's opt-in prepared route cache."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from cellerator.torch.execution import SparseSupportModule


class _BoundParameter:
    def __init__(self, pointer=0x1000):
        self.device = torch.device("cuda:2")
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class _SupportIdentity:
    def __init__(self):
        self.epoch = 7

    def snapshot(self):
        return {
            "structure_epoch": self.epoch,
            "base": {
                "structure_epoch": self.epoch,
                "relation_id": 101,
                "structure_id": 202,
            },
        }


class _FakeRouter:
    def __init__(self, *args):
        self.args = args
        self.queries = []

    def query(self, entity_ids, *identity):
        self.queries.append((entity_ids, identity))
        return "offsets", "mechanisms"


class _FakeNative:
    def __init__(self):
        self.prepared = []

    def PreparedEntityMechanismRouting(self, *args):
        router = _FakeRouter(*args)
        self.prepared.append(router)
        return router


class PreparedModuleRouteStrategyTests(unittest.TestCase):
    def _module(self):
        module = SparseSupportModule.__new__(SparseSupportModule)
        torch.nn.Module.__init__(module)
        module.parameters_flat = _BoundParameter()
        module._native = object()
        module._torch_native = _FakeNative()
        module._support_index = _SupportIdentity()
        module.structure_epoch = 7
        module._blocks = {"transition_A": SimpleNamespace(shape=(3, 4, 4))}
        module._mechanism_route_generation = 0
        module._mechanism_route_postings = ((2, 1, 1), (), (0, 2))
        module._prepared_mechanism_route_cache = None
        module._mechanism_route_relation_identity = None
        module._mechanism_route_relation_epoch = -1
        module.application_state = {}
        module._refresh_mechanism_route_postings = lambda: module.route_generation
        module.route_generation = 0
        return module

    def test_activity_only_queries_reuse_one_index_and_python_remains_default(self):
        module = self._module()
        module.parameters_flat = torch.nn.Parameter(torch.zeros(1))
        default_offsets, default_mechanisms = module.affected_mechanism_workset(
            torch.tensor([[0, 1, -1]], dtype=torch.int64))
        self.assertEqual(default_offsets.tolist(), [0, 2])
        self.assertEqual(default_mechanisms.tolist(), [1, 2])
        self.assertEqual(module._torch_native.prepared, [])

        del module.parameters_flat
        module.parameters_flat = _BoundParameter()
        first_ids = torch.tensor([[0, 1]], dtype=torch.int64)
        second_ids = torch.tensor([[2, 2]], dtype=torch.int64)

        first = module.affected_mechanism_workset(first_ids, strategy="prepared")
        second = module.affected_mechanism_workset(second_ids, strategy="prepared")
        self.assertEqual(first, ("offsets", "mechanisms"))
        self.assertEqual(second, ("offsets", "mechanisms"))
        self.assertEqual(len(module._torch_native.prepared), 1)
        self.assertEqual(len(module._torch_native.prepared[0].queries), 2)
        self.assertEqual(module.application_state["prepared_route_prepare_count"], 1)
        self.assertEqual(module.application_state["prepared_route_count_syncs"], 2)

        with self.assertRaisesRegex(ValueError, "strategy"):
            module.affected_mechanism_workset(first_ids, strategy="planner")

    def test_route_structure_and_binding_changes_rebuild_only_the_projection(self):
        module = self._module()
        entity_ids = torch.tensor([[0, 1]], dtype=torch.int64)
        module.affected_mechanism_workset(entity_ids, strategy="prepared")
        initial = module._torch_native.prepared[-1]

        module.route_generation = 1
        module._mechanism_route_postings = ((0,), (2,), ())
        module.affected_mechanism_workset(entity_ids, strategy="prepared")
        after_value = module._torch_native.prepared[-1]
        self.assertIsNot(after_value, initial)
        self.assertEqual(after_value.args[0], ((0,), (2,), ()))

        module.structure_epoch = 8
        module._support_index.epoch = 8
        module._invalidate_prepared_mechanism_routes(clear_relation=True)
        module.affected_mechanism_workset(entity_ids, strategy="prepared")
        after_structure = module._torch_native.prepared[-1]
        self.assertIsNot(after_structure, after_value)
        self.assertEqual(after_structure.args[6], 8)

        module.parameters_flat = _BoundParameter(pointer=0x2000)
        module.affected_mechanism_workset(entity_ids, strategy="prepared")
        after_binding = module._torch_native.prepared[-1]
        self.assertIsNot(after_binding, after_structure)
        self.assertEqual(after_binding.args[8], 2)

        self.assertEqual(module.application_state["prepared_route_prepare_count"], 4)


if __name__ == "__main__":
    unittest.main()
