#!/usr/bin/env python3
"""Independently replay the frozen CE-GEO baseline under a CUDA gate lease."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "bench/ce_geo/evidence/baseline/baseline.json"

HOST_TARGETS = (
    "cellPackExecutionImageV2Test",
    "celleratorBuiltinCatalogTest",
    "celleratorLivePlannerFeaturesTest",
    "celleratorExecutableCoreIntegrationTest",
)

CUDA_TARGETS = (
    "cellPackPersistentPackingPayloadTest",
    "cellPackExecutionImageV2DeviceTest",
    "celleratorOpaqueExecutionArtifactTest",
    "celleratorExecutionSessionTest",
    "celleratorRowMaskedN1CandidateTest",
    "celleratorCsrFallbackCandidateTest",
    "celleratorCusparseCsrCandidateTest",
    "celleratorFeatureMajorSmallNCandidateTest",
    "celleratorTransposeBackwardCandidateTest",
)

# These are the reviewed post-golden source states. The golden is historical
# evidence and is intentionally not rewritten. A seventh drift, or a further
# change to one of these six files, is an unreviewed baseline change.
REVIEWED_SOURCE_DRIFTS = {
    "include/Cellerator/compute/operation/builtin_catalog.hh":
        "08c2c4c9f510e39620957dd0078df5ef6fc13dbf45418dfba7efab7223ae3993",
    "include/Cellerator/execution/program.hh":
        "2c2232bdcc55c62ab91eac5eba958f58870182634b57188cb6042ef97efb0117",
    "include/Cellerator/geometry/persistence/execution_image_v2.hh":
        "efbe675cc85850b1061cd1c694c16ded8d6655e0d826958309476ce109d7aadc",
    "src/compute/operation/builtin_catalog.cc":
        "26bc5bf54fb9df77a643f77c7fe554fd7ea9287192d7dcc1edd281046a844172",
    "src/execution/program.cc":
        "0114c938375639014dfccf877eafffa137962698e3f588fa7556546dc1a60c19",
    "src/geometry/persistence/execution_image_v2.cc":
        "479a014daee998709078e40134e28fac9e5c34c99bcf995d9960a7801dda8359",
}

EXPECTED_WIRE_CONTRACTS = {
    "CPK1": {
        "kind": "0x43504b31",
        "schema_version": 1,
        "alignment": 64,
        "fixture_definition":
            "tests/geometry/persistent_packing_payload_test.cu",
        "fixture_definition_sha256":
            "3548f7eb4cb979dcc1a3868ae31042ca963e47cda13d2d7aaf64be70fd7391db",
        "preservation": "wire bytes and v1 semantics remain unchanged",
    },
    "CPE2": {
        "kind": "0x43504532",
        "schema_version": 2,
        "alignment": 64,
        "header_bytes": 256,
        "section_entry_bytes": 64,
        "projection_entry_bytes": 64,
        "fixture_definition":
            "tests/geometry/persistence/execution_image_v2_test.cc",
        "fixture_definition_sha256":
            "9c67cc1876739de3f2cd3cd2bc53551c0d5012b9ee13af267103923f2b4f7c6c",
    },
}

EXPECTED_CATALOG = {
    "builtin_candidate_count": 5,
    "normal_catalog_excludes_historical_dense_fragment": True,
    "program_and_session_cuda_commands_recorded": True,
}

EXPECTED_NEGATIVE_CONTROL = {
    "rows": 512,
    "features": 32738,
    "logical_edges": 433808,
    "qualified_fragment_tiles": 0,
    "maximum_tile_nnz": 106,
    "maximum_tile_density_less_than": 0.5,
    "whole_structure_padding_ratio_greater_than": 38.0,
    "disposition": "not_promoted",
}

EXPECTED_HISTORICAL_DENSE_FRAGMENT = {
    "source": "tests/tensor_core/v100_dense_fragment_candidate_test.cu",
    "source_sha256":
        "199ff2ae371219004954f4d75d9a35f2792491bd92eb7aa42cc4d963501b1a7f",
    "status": "preserved_negative_control_outside_normal_catalog",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("[CE-GEO-100] " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        fail(f"command failed with status {completed.returncode}: {command}")
    return completed


def invariant_subset(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: value.get(key) for key in keys}


def validate_golden(golden: dict[str, Any]) -> None:
    if golden.get("schema_version") != 1 or golden.get("task_id") != "CE-GEO-03":
        fail("baseline golden header changed")

    wire = golden.get("wire_contracts")
    if not isinstance(wire, dict):
        fail("baseline golden has no wire contracts")
    for name, expected in EXPECTED_WIRE_CONTRACTS.items():
        actual = wire.get(name)
        if not isinstance(actual, dict):
            fail(f"baseline golden has no {name} contract")
        if invariant_subset(actual, tuple(expected)) != expected:
            fail(f"baseline golden {name} invariants changed")

    if golden.get("catalog_program_session") != EXPECTED_CATALOG:
        fail("baseline golden catalog/program/session invariants changed")
    if golden.get("pbmc3k_negative_control") != EXPECTED_NEGATIVE_CONTROL:
        fail("baseline golden PBMC3K negative-control invariants changed")

    historical = golden.get("historical_dense_fragment")
    historical_changed = not isinstance(historical, dict) or (
        invariant_subset(historical, tuple(EXPECTED_HISTORICAL_DENSE_FRAGMENT))
        != EXPECTED_HISTORICAL_DENSE_FRAGMENT
    )
    if historical_changed:
        fail("baseline golden historical dense-fragment evidence changed")

    cuda = golden.get("cuda_regression_commands")
    if not isinstance(cuda, dict):
        fail("baseline golden has no CUDA command inventory")
    recorded = tuple(
        pathlib.Path(command[0]).name
        for command in cuda.get("commands", [])
        if isinstance(command, list) and command
    )
    if recorded != CUDA_TARGETS:
        fail(f"baseline golden CUDA inventory changed: {recorded}")

    host = golden.get("host_safe_results")
    if not isinstance(host, list) or len(host) < len(HOST_TARGETS):
        fail("baseline golden host result inventory is incomplete")
    recorded_host = tuple(pathlib.Path(item["argv"][0]).name
                          for item in host[:len(HOST_TARGETS)])
    if recorded_host != HOST_TARGETS:
        fail(f"baseline golden host inventory changed: {recorded_host}")


def validate_pinned_sources(golden: dict[str, Any]) -> None:
    pinned = golden.get("pinned_source_sha256")
    if not isinstance(pinned, dict):
        fail("baseline golden has no pinned source hashes")

    drifts: dict[str, tuple[str, str]] = {}
    for relative, expected in pinned.items():
        path = ROOT / relative
        if not path.is_file():
            fail(f"missing pinned baseline source: {relative}")
        actual = sha256(path)
        if actual != expected:
            drifts[relative] = (expected, actual)

    if set(drifts) != set(REVIEWED_SOURCE_DRIFTS):
        fail("unreviewed pinned source drift set: "
             + json.dumps(drifts, sort_keys=True))
    for relative, reviewed_hash in REVIEWED_SOURCE_DRIFTS.items():
        old_hash, actual_hash = drifts[relative]
        if actual_hash != reviewed_hash:
            fail(f"reviewed source drift changed again: {relative}: "
                 f"golden={old_hash} reviewed={reviewed_hash} actual={actual_hash}")
        print(f"[CE-GEO-100] pinned source drift: {relative} "
              f"golden={old_hash} current={actual_hash}")


def rebuild_targets(build: pathlib.Path) -> None:
    started = time.time_ns()
    jobs = max(1, os.cpu_count() or 1)
    run(["cmake", "--build", str(build), "--target",
         *HOST_TARGETS, *CUDA_TARGETS, "--clean-first", "-j", str(jobs)])
    missing = []
    stale = []
    for target in (*HOST_TARGETS, *CUDA_TARGETS):
        binary = build / target
        if not binary.is_file():
            missing.append(str(binary))
        elif binary.stat().st_mtime_ns < started:
            stale.append(str(binary))
    if missing:
        fail("missing rebuilt baseline targets: " + ", ".join(missing))
    if stale:
        fail("stale baseline targets after clean rebuild: " + ", ".join(stale))


def run_regressions(build: pathlib.Path, golden: dict[str, Any]) -> None:
    golden_host = golden["host_safe_results"]
    for index, target in enumerate(HOST_TARGETS):
        completed = run([str(build / target)])
        expected = golden_host[index]
        if completed.stdout != expected.get("stdout", ""):
            fail(f"host golden stdout changed for {target}: "
                 f"expected={expected.get('stdout', '')!r} "
                 f"actual={completed.stdout!r}")
        if completed.stderr != expected.get("stderr", ""):
            fail(f"host golden stderr changed for {target}")

    # This live decision test is separate from the four binary host baselines.
    run([sys.executable,
         "tests/ce_geo/baseline/test_dense_fragment_negative_control.py"])

    for target in CUDA_TARGETS:
        run([str(build / target)])

    experimental = build / "v100DenseFragmentCandidateTest"
    if experimental.exists():
        run([str(experimental)])
        print("[CE-GEO-100] historical experimental CUDA target: ran")
    else:
        print("[CE-GEO-100] historical experimental CUDA target: "
              "missing from the CMake build; not run and not counted among "
              "the nine recorded CUDA baselines")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    if not GOLDEN.is_file():
        fail(f"missing baseline golden: {GOLDEN}")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    validate_golden(golden)
    validate_pinned_sources(golden)

    build = (ROOT / arguments.build).resolve()
    if not (build / "CMakeCache.txt").is_file():
        fail(f"baseline build is not configured: {build}")
    rebuild_targets(build)
    run_regressions(build, golden)

    print("CE-GEO-100 baseline golden regression passed: "
          "4 host binaries, 9 CUDA binaries, 1 negative-control decision, "
          "6 reviewed source drifts")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError,
            json.JSONDecodeError) as error:
        print(f"CE-GEO-100 baseline golden regression failed: {error}",
              file=sys.stderr)
        raise SystemExit(1)
