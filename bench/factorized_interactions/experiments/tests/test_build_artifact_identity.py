"""CPU checks for KDA build artifacts linking the prepared control and candidate."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from adapter import _build_candidate  # noqa: E402
from common import AdapterError, candidate_tree_identity, checked_build  # noqa: E402


BUILDER_SOURCE = r'''from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source_manifest(root):
    rows = []
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(current) / name
            rows.append({"path": path.relative_to(root).as_posix(),
                         "bytes": path.stat().st_size, "sha256": sha(path)})
    rows.sort(key=lambda row: row["path"])
    digest = hashlib.sha256()
    for row in rows:
        path_bytes = row["path"].encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "little"))
        digest.update(path_bytes)
        digest.update(row["bytes"].to_bytes(8, "little"))
        digest.update(bytes.fromhex(row["sha256"]))
    return rows, digest.hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--corrupt", choices=("none", "source", "candidate_manifest_link",
                                           "candidate_binary_link", "baseline_runner_hash"),
                    default="none")
args = parser.parse_args()
root = args.build / "native_build"
binaries = root / "cmake"
binaries.mkdir(parents=True)
candidate_runner = binaries / "candidate-runner"
baseline_runner = binaries / "baseline-runner"
library = binaries / "libfactorized.a"
candidate_runner.write_bytes(b"candidate runner bytes")
baseline_runner.write_bytes(b"baseline runner bytes")
library.write_bytes(b"shared native library bytes")
source_files, source_hash = source_manifest(args.source)

def record(path):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha(path)}

candidate_manifest = root / "factorized_interactions_build_manifest.json"
candidate_data = {
    "schema": "cellerator.factorized_interactions.build.v1",
    "source_closure_sha256": source_hash, "source_files": source_files,
    "architecture": "sm_70", "fast_math": False,
    "binaries": {"runner": record(candidate_runner),
                 "factorized_interaction_cuda": record(library)},
}
candidate_manifest.write_text(json.dumps(candidate_data, sort_keys=True), encoding="utf-8")
candidate_link_hash = sha(candidate_manifest)
candidate_binary_hash = sha(candidate_runner)
baseline_data = {
    **candidate_data,
    "binaries": {"runner": record(baseline_runner),
                 "factorized_interaction_cuda": record(library)},
    "candidate_runner": {
        "manifest_path": candidate_manifest.relative_to(root).as_posix(),
        "manifest_sha256": candidate_link_hash,
        "binary_path": candidate_runner.relative_to(root).as_posix(),
        "binary_sha256": candidate_binary_hash,
    },
}
if args.corrupt == "source":
    baseline_data["source_closure_sha256"] = "0" * 64
elif args.corrupt == "candidate_manifest_link":
    baseline_data["candidate_runner"]["manifest_sha256"] = "0" * 64
elif args.corrupt == "candidate_binary_link":
    baseline_data["candidate_runner"]["binary_sha256"] = "0" * 64
elif args.corrupt == "baseline_runner_hash":
    baseline_data["binaries"]["runner"]["sha256"] = "0" * 64
baseline_manifest = root / "factorized_interactions_baseline_build_manifest.json"
baseline_manifest.write_text(json.dumps(baseline_data, sort_keys=True), encoding="utf-8")
'''


class PreparedBuildArtifactIdentityTests(unittest.TestCase):
    def _fixture(self, corruption: str = "none"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "snapshot"
        (source / "bench" / "factorized_interactions").mkdir(parents=True)
        (source / "CMakeLists.txt").write_text("project(test)\n", encoding="utf-8")
        (source / "bench" / "factorized_interactions" / "CMakeLists.txt").write_text(
            "project(test_bench)\n", encoding="utf-8")
        builder = source / "bench" / "factorized_interactions" / "build_fake.py"
        builder.write_text(BUILDER_SOURCE, encoding="utf-8")
        output = root / "kda-build"
        output.mkdir()
        snapshot_hash, _ = candidate_tree_identity(source)
        request = {
            "request_id": "build-1",
            "candidate": {"source_dir": str(source), "snapshot_sha256": snapshot_hash},
            "build": {"directory": str(output)},
        }
        variant_map = {"builder": {
            "trusted_script": str(builder),
            "snapshot_script_relative_path": "bench/factorized_interactions/build_fake.py",
            "argv": ["--source", "{source}", "--build", "{build}",
                     "--corrupt", corruption],
            "manifest_relative_path":
                "native_candidate/native_build/factorized_interactions_build_manifest.json",
        }}
        return request, variant_map, output

    def test_build_response_binds_baseline_candidate_and_shared_library(self) -> None:
        request, variant_map, output = self._fixture()
        result = _build_candidate(request, variant_map)
        self.assertEqual(result["status"], "ok")
        artifact_paths = {row["path"] for row in result["artifacts"]}
        self.assertIn(
            "native_candidate/native_build/factorized_interactions_baseline_build_manifest.json",
            artifact_paths,
        )
        self.assertIn(
            "native_candidate/native_build/cmake/baseline-runner", artifact_paths,
        )
        self.assertEqual(len(artifact_paths), 5)
        for row in result["artifacts"]:
            artifact = output / row["path"]
            self.assertEqual(row["sha256"], __import__("hashlib").sha256(
                artifact.read_bytes()).hexdigest())
        provenance = result["provenance"]
        for key in (
                "baseline_native_build_manifest_sha256", "baseline_runner_binary_sha256",
                "baseline_source_closure_sha256", "baseline_candidate_manifest_link_sha256",
                "baseline_candidate_runner_link_sha256"):
            self.assertRegex(provenance[key], r"^[0-9a-f]{64}$")
        build_root = output / "native_candidate" / "native_build"
        self.assertEqual(
            provenance["baseline_native_build_manifest_sha256"],
            __import__("hashlib").sha256(
                (build_root / "factorized_interactions_baseline_build_manifest.json").read_bytes()
            ).hexdigest(),
        )
        runner, manifest, _runner_hash = checked_build(
            request["candidate"],
            {"directory": str(output), "artifacts": result["artifacts"]},
            {"native_manifest_relative_path":
             "native_candidate/native_build/factorized_interactions_build_manifest.json"},
            {"snapshot_sha256": request["candidate"]["snapshot_sha256"],
             "snapshot_dir": request["candidate"]["source_dir"]},
        )
        self.assertEqual(manifest, build_root / "factorized_interactions_build_manifest.json")
        self.assertEqual(runner, build_root / "cmake" / "candidate-runner")

    def test_mismatched_baseline_source_closure_is_rejected(self) -> None:
        request, variant_map, _output = self._fixture("source")
        with self.assertRaisesRegex(AdapterError, "baseline build manifest source closure digest"):
            _build_candidate(request, variant_map)

    def test_mismatched_candidate_manifest_link_is_rejected(self) -> None:
        request, variant_map, _output = self._fixture("candidate_manifest_link")
        with self.assertRaisesRegex(AdapterError, "manifest link is stale or mismatched"):
            _build_candidate(request, variant_map)

    def test_mismatched_candidate_binary_link_is_rejected(self) -> None:
        request, variant_map, _output = self._fixture("candidate_binary_link")
        with self.assertRaisesRegex(AdapterError, "runner link is stale or mismatched"):
            _build_candidate(request, variant_map)

    def test_mismatched_baseline_runner_digest_is_rejected(self) -> None:
        request, variant_map, _output = self._fixture("baseline_runner_hash")
        with self.assertRaisesRegex(AdapterError, "baseline runner hash or size"):
            _build_candidate(request, variant_map)


if __name__ == "__main__":
    unittest.main()
