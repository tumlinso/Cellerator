#!/usr/bin/env python3
"""CPU-only tests for CE-ARCH-30 evidence manifests and generators."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import trace_tool
import validate_evidence


WORKLOADS = validate_evidence.MANIFESTS / "workloads.json"


class TraceToolTest(unittest.TestCase):
    def test_synthetic_trace_is_byte_deterministic(self) -> None:
        manifest = trace_tool.load_json(WORKLOADS)
        workload = trace_tool.find_workload(manifest, "scatac-modular-smoke")
        rows_a = list(trace_tool.rows_from_synthetic(workload["synthetic"]))
        rows_b = list(trace_tool.rows_from_synthetic(workload["synthetic"]))
        trace_a = trace_tool.compact_trace(
            workload["id"], rows_a, workload["synthetic"]["columns"],
            workload["axes"], {"test": True},
        )
        trace_b = trace_tool.compact_trace(
            workload["id"], rows_b, workload["synthetic"]["columns"],
            workload["axes"], {"test": True},
        )
        self.assertEqual(
            json.dumps(trace_a, sort_keys=True, separators=(",", ":")),
            json.dumps(trace_b, sort_keys=True, separators=(",", ":")),
        )
        trace_tool.validate_compact_trace(trace_a)

    def test_seed_changes_support(self) -> None:
        manifest = trace_tool.load_json(WORKLOADS)
        workload = trace_tool.find_workload(manifest, "scatac-modular-smoke")
        original = dict(workload["synthetic"])
        changed = dict(original)
        changed["seed"] += 1
        self.assertNotEqual(
            list(trace_tool.rows_from_synthetic(original)),
            list(trace_tool.rows_from_synthetic(changed)),
        )

    def test_matrix_market_is_byte_deterministic(self) -> None:
        rows = [[0, 4], [], [1, 3]]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.mtx"
            second = Path(directory) / "second.mtx"
            trace_tool.write_matrix_market(first, "deterministic", rows, 5)
            trace_tool.write_matrix_market(second, "deterministic", rows, 5)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(trace_tool.file_sha256(first), trace_tool.file_sha256(second))

    def test_compact_trace_rejects_checksum_tampering(self) -> None:
        trace = trace_tool.compact_trace(
            "tamper", [[1], [0, 2]], 3,
            {"row_axis": "row", "column_axis": "column"}, {"test": True},
        )
        trace["column_indices"][0] = 2
        with self.assertRaisesRegex(ValueError, "checksum"):
            trace_tool.validate_compact_trace(trace)

    def test_h5ad_support_extraction_is_deterministic(self) -> None:
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.h5ad"
            with h5py.File(path, "w") as handle:
                matrix = handle.create_group("X")
                matrix.attrs["shape"] = [4, 5]
                matrix.create_dataset("indptr", data=[0, 2, 3, 5, 6])
                matrix.create_dataset("indices", data=[0, 3, 1, 2, 4, 0])
            first = trace_tool.h5ad_support_rows(path, 3, 11)
            second = trace_tool.h5ad_support_rows(path, 3, 11)
            self.assertEqual(first, second)
            self.assertEqual(first[1], 5)


class PackageValidationTest(unittest.TestCase):
    def test_package_contracts_and_smoke_traces(self) -> None:
        result = validate_evidence.validate_package(False)
        self.assertEqual(result["status"], "valid")
        self.assertFalse(result["performance_run"])
        self.assertGreaterEqual(len(result["smoke_trace_ids"]), 3)
        self.assertEqual(len(result["representative_trace_ids"]), 2)

    def test_real_trace_features_are_checksum_pinned(self) -> None:
        sources = trace_tool.load_json(validate_evidence.MANIFESTS / "sources.json")
        trace_ids = validate_evidence.validate_representative_traces(sources)
        self.assertEqual(
            trace_ids,
            [
                "gse147520-local-x-support-r256-s7",
                "pbmc3k-raw-local-support-r512-s7",
            ],
        )

    def test_source_manifest_checksums_are_well_formed(self) -> None:
        sources = trace_tool.load_json(validate_evidence.MANIFESTS / "sources.json")
        for source in sources["sources"]:
            self.assertEqual(len(source["sha256"]), hashlib.sha256().digest_size * 2)


if __name__ == "__main__":
    unittest.main()
