#!/usr/bin/env python3
"""Capture one bounded bitmap-vs-postings GPU run with source identities."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import time


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(__file__).resolve().parents[4]
    sources = [
        pathlib.Path(__file__).with_name("bitmap_retrieval_bench.cu"),
        pathlib.Path(__file__).with_name("CMakeLists.txt"),
        pathlib.Path(__file__),
        pathlib.Path(__file__).with_name("gpu_job.json"),
        root / "src/compute/indexing/cuda_postings.cu",
        root / "include/Cellerator/compute/indexing/cuda_postings.cuh",
        root / "bench/sparse_support/evidence/cuda_indexing_v100_qualification_20261027.jsonl",
    ]
    command = [str(args.binary), "--samples", "21", "--iterations", "5"]
    result = subprocess.run(command, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr, end="")
        return result.returncode or 1
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(records) != 4:
        raise RuntimeError(f"expected four result records, received {len(records)}")
    for record in records:
        if record.get("correctness") != "scalar_oracle_exact_and_overflow_pass":
            raise RuntimeError("one or more GPU exactness checks failed")
    manifest = {
        "schema": "cellerator.sparse_support.bitmap_retrieval_run.v1",
        "created_unix_ns": time.time_ns(),
        "command": command,
        "binary": str(args.binary.resolve()),
        "binary_sha256": sha256(args.binary),
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in sources},
        "official_baseline_input_sha256": sha256(root / "bench/sparse_support/evidence/cuda_indexing_v100_qualification_20261027.jsonl"),
        "records": records,
        "device_runtime": "V100 SM70, CUDA 12.9 (runtime asserted in benchmark)",
        "timing": "paired CUDA event rounds; 3 warmups, 21 samples, 5 complete queries per candidate sample; candidate order rotates",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "results.jsonl").write_text(result.stdout)
    print(json.dumps({"output_dir": str(args.output_dir),
        "binary_sha256": manifest["binary_sha256"],
        "cases": len(records), "correctness": "exact"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
