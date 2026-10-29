#!/usr/bin/env python3
"""Run the frozen native discovery/gating baseline over v3 scale anchors.

This wrapper is intended to run only inside an admitted Device Scheduler job.
It preserves each case result separately and verifies fixture and binary
identities through run_cuda_indexing_bench.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "bench/sparse_support/evidence/experiments/scaled_sparse_campaign_20261027"
FIXTURES = Path("/tmp/cellerator_scaled_sparse_fixtures_887437fc")
BINARY = CAMPAIGN / "frozen_baseline/cellerator_sparse_support_cuda_indexing_bench"
RUNNER = ROOT / "bench/sparse_support/run_cuda_indexing_bench.py"
OUTPUT = CAMPAIGN / "native_baseline/run_001"
CASES = (
    ("small_v3", 96, 8, 64, "small"),
    ("large_ranked_v3", 262144, 32, 4096, "large-ranked"),
    ("large_events_v3", 1280000, 8, 20000, "large-events"),
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> int:
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        raise SystemExit("native CUDA benchmark requires Device Scheduler admission")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command_rows = []
    started = time.time()
    for case, edges, cells, entities, profile in CASES:
        result_path = OUTPUT / f"{case}.json"
        command = [
            sys.executable, str(RUNNER), "--binary", str(BINARY),
            "--fixture-dir", str(FIXTURES / profile), "--case-id", case,
            "--edges", str(edges), "--cells", str(cells), "--entities", str(entities),
            "--warmup", "5", "--samples", "21", "--iterations", "3",
            "--output", str(result_path),
        ]
        proc = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                              text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            (OUTPUT / f"{case}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
            (OUTPUT / f"{case}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
            raise RuntimeError(f"{case} failed with exit {proc.returncode}: {proc.stderr}")
        result = json.loads(proc.stdout)
        command_rows.append({
            "case_id": case, "profile": profile, "command": command,
            "result_path": str(result_path), "result_sha256": digest(result_path),
            "case_identity": result["input_identity"],
        })
    manifest = {
        "schema": "cellerator.sparse_support.scaled_native_baseline.v1",
        "started_unix_seconds": started,
        "completed_unix_seconds": time.time(),
        "scheduler_job_id": os.environ.get("DEVICE_SCHEDULER_JOB_ID"),
        "scheduler_attempt_id": os.environ.get("DEVICE_SCHEDULER_ATTEMPT_ID"),
        "admitted_marker": os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED"),
        "wrapper_sha256": digest(Path(__file__).resolve()),
        "runner_sha256": digest(RUNNER),
        "binary_sha256": digest(BINARY),
        "binary_path": str(BINARY),
        "cases": command_rows,
        "scope": "full-rescan versus reverse-posting discovery; prepared root-map full/dirty bitmap; bitmap workset construction; host index/projection/root-map setup and H2D setup reported separately",
        "limitations": [
            "Synthetic performance fixtures; not biological measurements.",
            "This is the serial reverse-posting baseline, frozen before the ordered-disjoint posting fastpath.",
            "Workset compaction uses the production bitmap-only API; the prepared direct-root map is used by full and dirty bitmap evaluation.",
        ],
    }
    manifest_path = OUTPUT / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT),
                      "manifest_sha256": digest(manifest_path),
                      "case_ids": [row["case_id"] for row in command_rows]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
