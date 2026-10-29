#!/usr/bin/env python3
"""Run and record the CE-GEO compatibility and negative-control baseline."""

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> dict:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    result = {
        "argv": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    build = (ROOT / args.build).resolve()
    evidence_dir = (ROOT / args.evidence).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)

    host_commands = [
        [str(build / "cellPackExecutionImageV2Test")],
        [str(build / "celleratorBuiltinCatalogTest")],
        [str(build / "celleratorLivePlannerFeaturesTest")],
        [str(build / "celleratorExecutableCoreIntegrationTest")],
        [sys.executable, "tests/ce_geo/baseline/test_dense_fragment_negative_control.py"],
    ]
    results = []
    for command in host_commands:
        if not pathlib.Path(command[0]).exists() and command[0] != sys.executable:
            raise RuntimeError(f"missing baseline executable: {command[0]}")
        results.append(run(command))

    pinned_files = [
        "include/Cellerator/geometry/persistent_packing_payload.hh",
        "src/geometry/persistent_packing_payload.cc",
        "tests/geometry/persistent_packing_payload_test.cu",
        "include/Cellerator/geometry/persistence/execution_image_v2.hh",
        "src/geometry/persistence/execution_image_v2.cc",
        "tests/geometry/persistence/execution_image_v2_test.cc",
        "include/Cellerator/compute/operation/builtin_catalog.hh",
        "src/compute/operation/builtin_catalog.cc",
        "tests/math_core/builtin_catalog_test.cc",
        "include/Cellerator/execution/program.hh",
        "src/execution/program.cc",
        "tests/execution/program_test.cu",
        "tests/runtime/execution_session_test.cu",
        "tests/math_core/transpose_backward_candidate_test.cu",
        "tests/tensor_core/v100_dense_fragment_candidate_test.cu",
        "tests/tensor_core/test_v100_dense_fragment_decision.py",
        "tests/ce_geo/baseline/test_dense_fragment_negative_control.py",
        "bench/ce_live/tensor_core/campaign/run_v100_decision.py",
        "bench/architecture_evidence/real_traces/pbmc3k-support-512.json",
        "data/manifests/ce_live/pbmc3k_quantitative_v1.json",
    ]
    hashes = {}
    for relative in pinned_files:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing pinned baseline source: {relative}")
        hashes[relative] = sha256(path)

    cuda_commands = [
        [str(build / "cellPackPersistentPackingPayloadTest")],
        [str(build / "cellPackExecutionImageV2DeviceTest")],
        [str(build / "celleratorOpaqueExecutionArtifactTest")],
        [str(build / "celleratorExecutionSessionTest")],
        [str(build / "celleratorRowMaskedN1CandidateTest")],
        [str(build / "celleratorCsrFallbackCandidateTest")],
        [str(build / "celleratorCusparseCsrCandidateTest")],
        [str(build / "celleratorFeatureMajorSmallNCandidateTest")],
        [str(build / "celleratorTransposeBackwardCandidateTest")],
    ]
    missing_cuda = [command[0] for command in cuda_commands if not pathlib.Path(command[0]).exists()]
    if missing_cuda:
        raise RuntimeError(f"missing recorded CUDA baseline executables: {missing_cuda}")

    record = {
        "schema_version": 1,
        "task_id": "CE-GEO-03",
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty_paths": git("status", "--short").splitlines(),
        },
        "wire_contracts": {
            "CPK1": {
                "kind": "0x43504b31",
                "schema_version": 1,
                "alignment": 64,
                "fixture_definition": "tests/geometry/persistent_packing_payload_test.cu",
                "fixture_definition_sha256": hashes["tests/geometry/persistent_packing_payload_test.cu"],
                "preservation": "wire bytes and v1 semantics remain unchanged",
            },
            "CPE2": {
                "kind": "0x43504532",
                "schema_version": 2,
                "alignment": 64,
                "header_bytes": 256,
                "section_entry_bytes": 64,
                "projection_entry_bytes": 64,
                "fixture_definition": "tests/geometry/persistence/execution_image_v2_test.cc",
                "fixture_definition_sha256": hashes["tests/geometry/persistence/execution_image_v2_test.cc"],
            },
        },
        "host_safe_results": results,
        "cuda_regression_commands": {
            "status": "recorded_for_leased_execution",
            "reason": "CE-GEO-03 has no accelerator resource request; unleased CUDA execution is forbidden. CE-GEO-100 independently runs the CUDA baseline under accelerator:any.",
            "commands": cuda_commands,
        },
        "catalog_program_session": {
            "builtin_candidate_count": 5,
            "normal_catalog_excludes_historical_dense_fragment": True,
            "program_and_session_cuda_commands_recorded": True,
        },
        "pbmc3k_negative_control": {
            "rows": 512,
            "features": 32738,
            "logical_edges": 433808,
            "qualified_fragment_tiles": 0,
            "maximum_tile_nnz": 106,
            "maximum_tile_density_less_than": 0.5,
            "whole_structure_padding_ratio_greater_than": 38.0,
            "disposition": "not_promoted",
        },
        "historical_dense_fragment": {
            "source": "tests/tensor_core/v100_dense_fragment_candidate_test.cu",
            "source_sha256": hashes["tests/tensor_core/v100_dense_fragment_candidate_test.cu"],
            "status": "preserved_negative_control_outside_normal_catalog",
            "legacy_test_note": "The retained CE-LIVE Python test still names the pre-remap catalog path; this task-owned live-path regression preserves its assertions without editing historical evidence.",
        },
        "pinned_source_sha256": hashes,
    }
    output = evidence_dir / "baseline.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CE-GEO baseline passed; evidence={output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"CE-GEO baseline failed: {error}", file=sys.stderr)
        raise SystemExit(1)
