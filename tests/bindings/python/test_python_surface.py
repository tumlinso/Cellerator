"""Public Python module routes and legacy sparse import preservation."""
import unittest
from torch import nn

from cellerator import nn as cellerator_nn
from cellerator import ops as cellerator_ops
from cellerator.torch import (
    execution, gating, indexing, interop, layout, nn as torch_nn, routing,
    structure, support,
)
from cellerator.torch.sparse_support import (
    GateValues as LegacyGateValues,
    RankedGeneSlice as LegacyRankedGeneSlice,
    SupportSpec as LegacySupportSpec,
    make_next_event_fixture as legacy_fixture,
)
from cellerator.torch.testing import make_next_event_fixture


class PythonSurfaceTest(unittest.TestCase):
    def test_public_modules_own_real_capabilities(self):
        self.assertTrue(cellerator_ops.product2_forward)
        self.assertTrue(issubclass(torch_nn.Product2Module, nn.Module))
        self.assertIs(cellerator_nn.Product2Module, torch_nn.Product2Module)
        self.assertIs(structure.SupportSpec, support.SupportSpec)
        self.assertIs(indexing.PerCellWorkset, structure.PerCellWorkset)
        self.assertTrue(execution.SparseSupportModule)
        self.assertTrue(routing.RankedGeneSlice)
        self.assertTrue(gating.learned_gates)
        self.assertTrue(layout.logical_edge_values)
        self.assertIsNotNone(interop.cuda)

    def test_legacy_sparse_imports_forward_to_new_owners(self):
        self.assertIs(LegacySupportSpec, structure.SupportSpec)
        self.assertIs(LegacyGateValues, structure.GateValues)
        self.assertIs(LegacyRankedGeneSlice, routing.RankedGeneSlice)
        self.assertIs(legacy_fixture, make_next_event_fixture)


if __name__ == "__main__":
    unittest.main()
