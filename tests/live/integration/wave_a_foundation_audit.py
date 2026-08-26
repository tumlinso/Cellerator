#!/usr/bin/env python3

import json
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]


def load_json(relative: str) -> dict:
    return json.loads((REPOSITORY / relative).read_text(encoding="utf-8"))


class WaveAFoundationAudit(unittest.TestCase):
    def test_relation_orientation_is_shared_by_all_contracts(self) -> None:
        orientation = (
            REPOSITORY / "docs/CE_LIVE_RELATION_ORIENTATION.md"
        ).read_text(encoding="utf-8")
        catalog = load_json("bench/ce_live/catalog/candidate_inventory_v1.json")
        tensor = load_json(
            "bench/ce_live/tensor_core/contract/"
            "v100_dense_fragment_candidate_v1.json"
        )

        self.assertIn(
            "feature or gene source -> row, cell, or module destination",
            orientation,
        )
        self.assertIn("same logical structure identity", orientation)
        self.assertEqual(
            "feature-source-to-row-destination", catalog["logical_relation"]
        )
        self.assertEqual(
            catalog["logical_relation"],
            tensor["semantic_contract"]["logical_relation"],
        )
        self.assertTrue(
            tensor["semantic_contract"][
                "structure_identity_shared_with_other_projections"
            ]
        )

    def test_fixture_remains_computational_and_checksum_pinned(self) -> None:
        fixture_doc = (
            REPOSITORY / "docs/CE_LIVE_QUANTITATIVE_FIXTURE.md"
        ).read_text(encoding="utf-8")
        manifest = load_json(
            "data/manifests/ce_live/pbmc3k_quantitative_v1.json"
        )

        self.assertIn("computational fixture", fixture_doc)
        self.assertIn("not evidence for a biological claim", fixture_doc)
        self.assertEqual(64, len(manifest["source"]["sha256"]))
        self.assertEqual(
            "observations_by_features", manifest["source"]["stored_orientation"]
        )
        self.assertEqual(2, len(manifest["generations"]))

    def test_inventory_and_tensor_lane_are_activation_inputs_only(self) -> None:
        catalog = load_json("bench/ce_live/catalog/candidate_inventory_v1.json")
        tensor = load_json(
            "bench/ce_live/tensor_core/contract/"
            "v100_dense_fragment_candidate_v1.json"
        )

        self.assertEqual("audit-only-not-runtime-abi", catalog["status"])
        self.assertGreaterEqual(len(catalog["candidates"]), 5)
        self.assertTrue(
            all(
                candidate["activation"] == "implemented-unregistered-by-default"
                for candidate in catalog["candidates"]
            )
        )
        self.assertEqual(
            "design-only-unimplemented-unregistered", tensor["status"]
        )
        self.assertFalse(tensor["qualification"]["global_default"])

    def test_readiness_is_runtime_only_and_native_build_is_torch_optional(self) -> None:
        readiness = (
            REPOSITORY / "include/Cellerator/runtime/value_readiness.cuh"
        ).read_text(encoding="utf-8")
        cmake = (REPOSITORY / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("Runtime-only readiness", readiness)
        self.assertIn("never synchronizes", readiness)
        self.assertIn("src/runtime/value_readiness.cu", cmake)
        self.assertIn("celleratorValueReadinessTest", cmake)
        self.assertIn(
            'option(CELLERATOR_ENABLE_TORCH_MODELS '
            '"Enable the CelleraTorch compatibility component targets" OFF)',
            cmake,
        )


if __name__ == "__main__":
    unittest.main()
