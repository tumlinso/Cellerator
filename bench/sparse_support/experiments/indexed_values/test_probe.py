from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


PROBE_PATH = Path(__file__).with_name("probe.py")
SPEC = importlib.util.spec_from_file_location("indexed_values_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(probe)


class _Support:
    edge_ids = tuple(range(17))


class _FakeModule:
    def __init__(self, entities, sources, destinations, scorer):
        self._blocks = {"entity_embedding": entities, "scorer": scorer}
        self._source_endpoint_indices = sources
        self._destination_endpoint_indices = destinations
        self.support = _Support()

    def block(self, name):
        return self._blocks[name]


class IndexedValuesReferenceTest(unittest.TestCase):
    def test_index_copy_matches_full_value_vjp_and_preserves_dormant_candidates(self):
        torch.manual_seed(4)
        entities = torch.randn(12, 7, requires_grad=True)
        scorer = torch.randn(7, requires_grad=True)
        sources = torch.randint(0, 12, (17,), dtype=torch.int64)
        destinations = torch.randint(0, 12, (17,), dtype=torch.int64)
        full_module = _FakeModule(entities, sources, destinations, scorer)
        indexed_module = _FakeModule(entities, sources, destinations, scorer)

        primal = torch.tensor([2, 8], dtype=torch.int64)
        derivative = torch.tensor([2, 5, 8, 13], dtype=torch.int64)
        used = torch.unique(torch.cat((primal, derivative)), sorted=True)

        def full_values(module):
            embedding = module.block("entity_embedding")
            source = embedding.index_select(0, module._source_endpoint_indices)
            destination = embedding.index_select(0, module._destination_endpoint_indices)
            return (source * destination * module.block("scorer")).sum(-1) / (7 ** 0.5)

        expected = full_values(full_module)
        actual = probe.indexed_values(indexed_module, used)
        weights = torch.randn_like(expected)
        mask = torch.isin(torch.arange(expected.numel()), used)
        expected_grad = torch.autograd.grad((expected * weights * mask).sum(),
                                            (entities, scorer), retain_graph=True)
        actual_grad = torch.autograd.grad((actual * weights).sum(),
                                          (entities, scorer))
        torch.testing.assert_close(actual, expected * mask, rtol=0, atol=0)
        torch.testing.assert_close(actual_grad[0], expected_grad[0], rtol=2e-7, atol=2e-7)
        torch.testing.assert_close(actual_grad[1], expected_grad[1], rtol=2e-7, atol=2e-7)
        self.assertIn(5, used.tolist())  # derivative-only support must stay scored
        self.assertIn(13, used.tolist())


if __name__ == "__main__":
    unittest.main()
