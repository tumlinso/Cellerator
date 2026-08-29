#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import unittest

from validate_evidence import (
    COLD_PHASES,
    EVIDENCE_SCHEMA,
    EvidenceError,
    WARM_PHASES,
    validate_command_manifest,
    validate_evidence,
)


DIGEST = hashlib.sha256(b"ce-geo-test").hexdigest()


def manifest() -> dict:
    return {
        "schema": "CELLERATOR-CE-GEO-COMMAND/1",
        "campaign_id": "ce-geo-harness-test",
        "project_root": "/home/tumlinson/Cellerator",
        "commands": {
            "build": [["cmake", "--build", "build", "-j", "20"]],
            "correctness": [["./build/correctness"]],
            "measure": [["./build/benchmark", "--repeats", "5"]],
        },
        "methodology": {
            "warmups": 2,
            "repeats": 5,
            "maximum_mad_percent": 5.0,
            "cold_warm_separated": True,
            "correctness_before_measurement": True,
        },
        "resources": {
            "required_leases": ["benchmark", "gpu"],
            "benchmark_mutex": "bench/benchmark_mutex.hh",
            "run_without_leases": False,
        },
        "capture": {name: True for name in
                    ("source", "device", "topology", "toolchain", "build")},
        "required_phases": {
            "cold": sorted(COLD_PHASES),
            "warm": sorted(WARM_PHASES),
        },
    }


def evidence() -> dict:
    samples = [100.0, 101.0, 99.0, 100.0, 102.0]
    return {
        "schema": EVIDENCE_SCHEMA,
        "campaign_id": "ce-geo-harness-test",
        "command_manifest_sha256": DIGEST,
        "source": {
            "revision": "0123456789abcdef",
            "clean": True,
            "status_digest": DIGEST,
            "todo_revision": "2450",
            "submodule_revisions": {},
        },
        "device": {
            "uuid": "GPU-test",
            "name": "test device",
            "pci_bus_id": "0000:00:00.0",
            "performance_class": "nvidia-sm70",
            "driver_version": "test",
        },
        "topology": {
            "capture_command": "controller supplied fixture",
            "capture_sha256": DIGEST,
        },
        "toolchain": {
            "cxx": "test c++17",
            "cuda_toolkit": "test",
            "nvcc": "test",
            "cmake": "test",
        },
        "build": {
            "mode": "Release",
            "architecture": "sm_70",
            "binary_sha256": DIGEST,
            "cmake_cache_sha256": DIGEST,
        },
        "command": {
            "argv": ["./build/benchmark", "--repeats", "5"],
            "cwd": "/immutable/Cellerator",
            "environment_digest": DIGEST,
        },
        "controller": {
            "acquired_leases": {"benchmark": "lease-b", "gpu": "lease-g"},
            "benchmark_mutex_acquired": True,
        },
        "methodology": {"warmups": 2, "repeats": 5, "clock": "cuda_event"},
        "phase_samples_ns": {
            "cold": {phase: [1.0] * 5 for phase in COLD_PHASES},
            "warm": {phase: [1.0] * 5 for phase in WARM_PHASES},
        },
        "complete_samples_ns": samples,
        "correctness": {
            "passed": True,
            "digest": DIGEST,
            "numerical_error": {"max_abs": 0.0, "relative_l2": 0.0},
        },
        "contamination": {"detected": False, "reasons": [], "attempt": 1},
        "summary": {
            "median_complete_ns": 100.0,
            "mad_percent": 1.0,
            "accepted": True,
        },
    }


class HarnessContractTest(unittest.TestCase):
    def test_valid_manifest_and_evidence(self) -> None:
        self.assertEqual(validate_command_manifest(manifest())["status"], "valid")
        result = validate_evidence(evidence(), manifest())
        self.assertTrue(result["accepted"])
        self.assertFalse(result["performance_run"])

    def test_unleased_manifest_is_rejected(self) -> None:
        value = manifest()
        value["resources"]["required_leases"] = ["gpu"]
        with self.assertRaisesRegex(EvidenceError, "leases"):
            validate_command_manifest(value)

    def test_dirty_source_is_rejected(self) -> None:
        value = evidence()
        value["source"]["clean"] = False
        with self.assertRaisesRegex(EvidenceError, "clean"):
            validate_evidence(value, manifest())

    def test_missing_phase_is_rejected(self) -> None:
        value = evidence()
        del value["phase_samples_ns"]["warm"]["kernel"]
        with self.assertRaisesRegex(EvidenceError, "phase contract"):
            validate_evidence(value, manifest())

    def test_spread_is_recomputed_and_fail_closed(self) -> None:
        value = evidence()
        value["summary"]["mad_percent"] = 0.0
        with self.assertRaisesRegex(EvidenceError, "not derived"):
            validate_evidence(value, manifest())

    def test_contamination_cannot_be_accepted(self) -> None:
        value = evidence()
        value["contamination"] = {
            "detected": True,
            "reasons": ["overlapping gpu process"],
            "attempt": 1,
        }
        with self.assertRaisesRegex(EvidenceError, "accepted disposition"):
            validate_evidence(value, manifest())


if __name__ == "__main__":
    unittest.main()
