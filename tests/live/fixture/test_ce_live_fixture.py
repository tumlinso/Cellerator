#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPT = REPOSITORY / "scripts" / "ce_live_fixture.py"
SPEC = importlib.util.spec_from_file_location("ce_live_fixture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


class QuantitativeFixtureTest(unittest.TestCase):
    def test_selection_matches_structural_fixture(self) -> None:
        rows = FIXTURE.choose_rows(2700, 512, 7)
        self.assertEqual(512, len(rows))
        self.assertEqual([4, 13, 18, 21], rows[:4])
        self.assertEqual([2674, 2676, 2684, 2695], rows[-4:])

    def test_tiny_fixture_has_independent_forward_and_transpose_referees(self) -> None:
        FIXTURE.verify_smoke(
            Path(__file__).with_name("tiny_quantitative_fixture_v1.json")
        )

    def test_local_source_reproduces_manifest_and_extract(self) -> None:
        source = REPOSITORY / "data/test/reference/pbmc3k_raw.h5ad"
        if not source.exists():
            self.skipTest("local checksum-pinned PBMC3K source is unavailable")
        manifest_path = REPOSITORY / "data/manifests/ce_live/pbmc3k_quantitative_v1.json"
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = FIXTURE.load_source(source)
        FIXTURE.verify_manifest(expected, actual)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture.npz"
            FIXTURE.write_npz(output, source, actual)
            import numpy as np

            with np.load(output) as extracted:
                self.assertEqual((513,), extracted["indptr"].shape)
                self.assertEqual(433808, len(extracted["indices"]))
                self.assertEqual(
                    expected["generations"][1]["values_sha256"],
                    FIXTURE.array_digest(extracted["generation_2_values"], "<f4"),
                )


if __name__ == "__main__":
    unittest.main()
