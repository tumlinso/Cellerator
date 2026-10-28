#!/usr/bin/env python3
"""Run and bind the CSR-union bitmap comparison to frozen source and fixture IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import time


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def verify_descriptor_set(root: pathlib.Path, descriptors: dict[str, object]) -> None:
    for name, descriptor in descriptors.items():
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"invalid array descriptor for {name}")
        filename = descriptor.get("file")
        if not isinstance(filename, str) or pathlib.Path(filename).name != filename:
            raise RuntimeError(f"invalid array filename for {name}")
        path = root / filename
        if not path.is_file() or path.stat().st_size != descriptor.get("nbytes"):
            raise RuntimeError(f"array size mismatch for {name}: {path}")
        if sha256(path) != descriptor.get("sha256"):
            raise RuntimeError(f"array digest mismatch for {name}: {path}")


def verify_fixture(input_root: pathlib.Path, manifest: dict[str, object]) -> None:
    if manifest.get("schema") != "cellerator.csr_union_bitmap_fixture.v1":
        raise RuntimeError("unexpected fixture manifest schema")
    metadata = manifest.get("fixture_metadata")
    if not isinstance(metadata, dict) or (
        metadata.get("edges"), metadata.get("entities"), metadata.get("cells")
    ) != (1_000_000, 20_000, 32):
        raise RuntimeError("fixture dimensions do not match the frozen benchmark")
    support = manifest.get("support_arrays")
    cases = manifest.get("query_cases")
    if not isinstance(support, dict) or not isinstance(cases, dict):
        raise RuntimeError("fixture manifest is missing support arrays or query cases")
    verify_descriptor_set(input_root / "support", support)
    for case, descriptors in cases.items():
        if not isinstance(descriptors, dict):
            raise RuntimeError(f"invalid descriptors for query case {case}")
        verify_descriptor_set(input_root / "queries" / case, descriptors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--input-root", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    binary = args.binary.resolve()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.jsonl"
    stderr_path = output_dir / "stderr.txt"
    if manifest_path.exists() or results_path.exists() or stderr_path.exists():
        parser.error("refusing to overwrite evidence")
    fixture_manifest_path = input_root / "manifest.json"
    fixture_manifest = json.loads(fixture_manifest_path.read_text())
    verify_fixture(input_root, fixture_manifest)
    root = pathlib.Path(__file__).resolve().parents[4]
    source_paths = [
        pathlib.Path(__file__).with_name("csr_union_bitmap_bench.cu"),
        pathlib.Path(__file__).with_name("CMakeLists.txt"),
        pathlib.Path(__file__).with_name("prepare_fixture.py"),
        pathlib.Path(__file__).with_name("run_experiment.py"),
        pathlib.Path(__file__).with_name("gpu_job.json"),
        root / "bench/sparse_support/scaled_fixtures.py",
        root / "include/Cellerator/compute/indexing/cuda_postings.cuh",
        root / "include/Cellerator/compute/indexing/reverse_postings.hh",
        root / "include/Cellerator/compute/support/logical_support.hh",
        root / "src/indexing/cuda_postings.cu",
        root / "src/indexing/reverse_postings.cc",
        root / "src/support/logical_support.cc",
    ]
    command = [
        str(binary), "--input-root", str(input_root),
        "--warmups", str(args.warmups), "--samples", str(args.samples),
    ]
    manifest = {
        "schema": "cellerator.csr_union_bitmap_run.v1",
        "state": "running",
        "started_unix_ns": time.time_ns(),
        "command": command,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "input_root": str(input_root),
        "fixture_manifest_sha256": sha256(fixture_manifest_path),
        "fixture_generator_sha256": fixture_manifest["generator_sha256"],
        "topology_sha256": fixture_manifest["fixture_metadata"]["topology_sha256"],
        "query_cases": sorted(fixture_manifest["query_cases"]),
        "source_sha256": {
            str(path.relative_to(root)): sha256(path) for path in source_paths
        },
        "results": [],
        "completed_cases": [],
    }
    write_json(manifest_path, manifest)

    failure = None
    with results_path.open("x", encoding="utf-8") as results:
        process = subprocess.Popen(command, cwd=root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                record = json.loads(line)
                if record.get("correctness") != "cpu_oracle_exact":
                    raise RuntimeError("benchmark emitted a result without exact oracle validation")
                if record["case_id"] in manifest["completed_cases"]:
                    raise RuntimeError("duplicate result case")
                if record["candidate_required_total"] > record["raw_posting_upper_bound"]:
                    raise RuntimeError("reported candidate count exceeds its safe capacity")
                results.write(json.dumps(record, sort_keys=True) + "\n")
                results.flush()
                manifest["results"].append(record["case_id"])
                manifest["completed_cases"].append(record["case_id"])
                write_json(manifest_path, manifest)
        except BaseException as error:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            failure = f"{type(error).__name__}: {error}"
        stderr = process.stderr.read() if process.stderr is not None else ""
        returncode = process.wait()

    stderr_path.write_text(stderr)
    setup = None
    for line in stderr.splitlines():
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "setup_ms" in parsed:
            setup = parsed["setup_ms"]
    manifest["returncode"] = returncode
    manifest["setup_ms"] = setup
    if failure is not None:
        manifest["failure"] = failure
    manifest["completed_unix_ns"] = time.time_ns()
    manifest["state"] = "succeeded" if returncode == 0 and failure is None else "failed"
    write_json(manifest_path, manifest)
    print(json.dumps({
        "output_dir": str(output_dir),
        "binary_sha256": manifest["binary_sha256"],
        "returncode": returncode,
        "completed_cases": manifest["completed_cases"],
    }, sort_keys=True))
    return returncode if returncode != 0 else (1 if failure is not None else 0)


if __name__ == "__main__":
    raise SystemExit(main())
