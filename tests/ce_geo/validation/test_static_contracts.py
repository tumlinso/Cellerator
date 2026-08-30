#!/usr/bin/env python3
"""Focused adversarial tests for the CE-GEO static-contract auditor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/ce_geo/check_static_contracts.py"
SPEC = importlib.util.spec_from_file_location("check_static_contracts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDITOR
SPEC.loader.exec_module(AUDITOR)


class StaticContractAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.production = "src/geometry/compiler/owned.cc"
        self.csg1 = "src/geometry/persistence/semantic_geometry_image_v1.cc"
        self.provider = (
            "src/compute/architecture/providers/nvidia/sm70/owned.cu"
        )
        self.cpk1 = "include/Cellerator/geometry/persistence/cpk1.hh"
        self.cpe2 = "include/Cellerator/geometry/persistence/cpe2.hh"
        self.write(self.production, "int compile_geometry() { return 1; }\n")
        self.write(self.csg1, "int validate_semantic_image() { return 1; }\n")
        self.write(self.provider, "int physical_kernel() { return 1; }\n")
        self.write(self.cpk1, "#pragma once\nstruct cpk1 {};\n")
        self.write(self.cpe2, "#pragma once\nstruct cpe2 {};\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def digest(self, relative: str) -> str:
        return hashlib.sha256((self.root / relative).read_bytes()).hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign": "CE-GEO",
            "scope": "owned-production-only",
            "production": [
                {"path": self.production, "layer": "compiler"},
                {"path": self.csg1, "layer": "csg1"},
                {"path": self.provider, "layer": "physical_provider"},
            ],
            "protected_compatibility": [
                {"path": self.cpk1, "contract": "CPK1",
                 "sha256": self.digest(self.cpk1)},
                {"path": self.cpe2, "contract": "CPE2",
                 "sha256": self.digest(self.cpe2)},
            ],
        }

    def audit(self, manifest: dict[str, object] | None = None) -> list[str]:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest or self.manifest()), encoding="utf-8")
        failures, _, _ = AUDITOR.audit_manifest(path, self.root)
        return failures

    def assertViolation(self, needle: str,
                        manifest: dict[str, object] | None = None) -> None:
        failures = self.audit(manifest)
        self.assertTrue(any(needle in failure for failure in failures), failures)

    def test_accepts_closed_ce_geo_inventory_and_ignores_prose(self) -> None:
        self.write(self.production, """
            // std::vector and atomicAdd are diagnostic words, not code.
            const char *message = "--use_fast_math nvcuda::wmma";
            int compile_geometry() { return message[0] != 0; }
        """)
        self.assertEqual([], self.audit())

    def test_rejects_new_stl_ownership(self) -> None:
        self.write(self.production, "#include <vector>\nstd::vector<int> values;\n")
        self.assertViolation("new STL ownership std::vector")

    def test_rejects_wmma_leak_but_allows_physical_provider(self) -> None:
        self.write(self.production, "void f() { nvcuda::wmma::mma_sync(); }\n")
        self.assertViolation("WMMA leaked")
        self.write(self.production, "int f() { return 0; }\n")
        self.write(self.provider, "void f() { nvcuda::wmma::mma_sync(); }\n")
        self.assertEqual([], self.audit())

    def test_rejects_architecture_detail_in_csg1(self) -> None:
        self.write(self.csg1, "constexpr int warp_size = 32;\n")
        self.assertViolation("architecture detail leaked into portable CSG1")

    def test_rejects_fast_math_and_atomics(self) -> None:
        self.write(self.production, "float f(float x) { return __expf(x); }\n")
        self.assertViolation("fast-math")
        self.write(self.production, "void f(float *x) { atomicAdd(x, 1.0f); }\n")
        self.assertViolation("atomic operation")

    def test_rejects_global_fast_math_in_cmake_string(self) -> None:
        cmake = "src/compute/architecture/providers/nvidia/common/CMakeLists.txt"
        self.write(cmake, 'target_compile_options(provider PRIVATE "--use_fast_math")\n')
        manifest = self.manifest()
        manifest["production"] = [
            {"path": cmake, "layer": "physical_provider"}
        ]
        self.assertViolation("fast-math", manifest)

    def test_rejects_mutated_cpk1_or_cpe2(self) -> None:
        manifest = self.manifest()
        self.write(self.cpe2, "#pragma once\nstruct changed_cpe2 {};\n")
        self.assertViolation("frozen CPE2 sha256 mismatch", manifest)

    def test_rejects_broad_or_ce_ptr_ownership(self) -> None:
        manifest = self.manifest()
        manifest["production"] = [
            {"path": "src/runtime/broad.cc", "layer": "geometry"}
        ]
        self.assertViolation("broad production ownership", manifest)
        manifest = self.manifest()
        manifest["campaign"] = "CE-PTR"
        self.assertViolation("manifest must certify CE-GEO only", manifest)

    def test_rejects_duplicate_empty_and_incomplete_freeze_inventory(self) -> None:
        manifest = self.manifest()
        manifest["production"] = []
        self.assertViolation("nonempty array", manifest)
        manifest = self.manifest()
        manifest["production"] = [
            {"path": self.production, "layer": "compiler"},
            {"path": self.production, "layer": "compiler"},
        ]
        self.assertViolation("duplicate production path", manifest)
        manifest = self.manifest()
        manifest["protected_compatibility"] = [
            {"path": self.cpe2, "contract": "CPE2",
             "sha256": self.digest(self.cpe2)}
        ]
        self.assertViolation("pin both CPK1 and CPE2", manifest)


if __name__ == "__main__":
    unittest.main()
