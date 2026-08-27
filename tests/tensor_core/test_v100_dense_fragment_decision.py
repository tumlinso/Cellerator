#!/usr/bin/env python3

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "bench/ce_live/tensor_core/campaign/run_v100_decision.py"
SPEC = importlib.util.spec_from_file_location("v100_decision", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DenseFragmentDecisionTest(unittest.TestCase):
    def test_density_bucket_boundaries(self) -> None:
        self.assertEqual(MODULE.density_bucket(0), "empty")
        self.assertEqual(MODULE.density_bucket(1), "low")
        self.assertEqual(MODULE.density_bucket(63), "low")
        self.assertEqual(MODULE.density_bucket(64), "medium")
        self.assertEqual(MODULE.density_bucket(127), "medium")
        self.assertEqual(MODULE.density_bucket(128), "high")
        self.assertEqual(MODULE.density_bucket(191), "high")
        self.assertEqual(MODULE.density_bucket(192), "near_dense")
        self.assertEqual(MODULE.density_bucket(256), "near_dense")

    def test_checksum_pinned_pbmc3k_has_no_qualified_fragment(self) -> None:
        trace = json.loads((ROOT /
            "bench/architecture_evidence/real_traces/pbmc3k-support-512.json"
        ).read_text())
        manifest = json.loads((ROOT /
            "data/manifests/ce_live/pbmc3k_quantitative_v1.json").read_text())
        result = MODULE.classify(trace, manifest)
        self.assertEqual(result["rows"], 512)
        self.assertEqual(result["features"], 32738)
        self.assertEqual(result["logical_edges"], 433808)
        self.assertEqual(result["qualified_fragment_tiles"], 0)
        self.assertEqual(result["maximum_tile_nnz"], 106)
        self.assertLess(result["maximum_tile_density"], 0.5)
        self.assertGreater(result["whole_structure_padding_ratio"], 38.0)

    def test_repo_catalog_does_not_promote_rejected_candidate(self) -> None:
        catalog = (ROOT /
            "src/compute/math/operation_core/builtin_catalog.cc").read_text()
        self.assertNotIn("v100_dense_fragment_candidate", catalog)
        self.assertNotIn("v100-wmma-dense-fragment-f16-f32", catalog)


if __name__ == "__main__":
    unittest.main()
