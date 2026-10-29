#!/usr/bin/env python3
"""Live-path regression for the preserved V100 dense-fragment decision."""

import importlib.util
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "bench/ce_live/tensor_core/campaign/run_v100_decision.py"
SPEC = importlib.util.spec_from_file_location("ce_geo_v100_decision", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DenseFragmentNegativeControl(unittest.TestCase):
    def test_pbmc3k_remains_unqualified(self) -> None:
        trace = json.loads((ROOT /
            "bench/architecture_evidence/real_traces/pbmc3k-support-512.json"
        ).read_text(encoding="utf-8"))
        manifest = json.loads((ROOT /
            "data/manifests/ce_live/pbmc3k_quantitative_v1.json"
        ).read_text(encoding="utf-8"))
        result = MODULE.classify(trace, manifest)
        self.assertEqual(result["rows"], 512)
        self.assertEqual(result["features"], 32738)
        self.assertEqual(result["logical_edges"], 433808)
        self.assertEqual(result["qualified_fragment_tiles"], 0)
        self.assertEqual(result["maximum_tile_nnz"], 106)
        self.assertLess(result["maximum_tile_density"], 0.5)
        self.assertGreater(result["whole_structure_padding_ratio"], 38.0)

    def test_live_catalog_does_not_promote_experiment(self) -> None:
        catalog = (ROOT /
            "src/compute/operation/builtin_catalog.cc").read_text(encoding="utf-8")
        self.assertNotIn("v100_dense_fragment_candidate", catalog)
        self.assertNotIn("v100-wmma-dense-fragment-f16-f32", catalog)


if __name__ == "__main__":
    unittest.main()
