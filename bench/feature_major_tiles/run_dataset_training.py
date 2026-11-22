#!/usr/bin/env python3
"""Run paired dataset training measurements inside an admitted GPU job."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys


ROUTES = ("tiled", "tiled-fused", "csr", "cusparse")
PHASES = ("gather", "forward", "backward_with_loss", "sgd_and_cast",
          "gather_forward_backward")
EPOCH_FIELDS = {
    "training_gpu": "training_gpu_ms_per_batch",
    "training_host": "training_host_ms_per_batch",
    "gather_forward_backward_gpu": "gather_forward_backward_gpu_ms_per_batch",
    "gather_forward_backward_host": "gather_forward_backward_host_ms_per_batch",
}
SEED = 613


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(record, route, fixture, args):
    expected = {"route": route, "oracle": "pass", "compute_capability": 70,
                "cells": fixture["cells"], "genes": fixture["resident_genes"],
                "targets": fixture["targets"], "tfs": fixture["regulators"],
                "edges": fixture["edges"], "epochs": args.epochs,
                "batches_per_epoch": (fixture["cells"] + 15) // 16}
    if any(record[field] != value for field, value in expected.items()):
        raise ValueError(f"Wrong benchmark configuration or failed oracle for {route}")
    for phase in PHASES:
        value = record["diagnostic_ms_per_batch"][phase]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Bad phase timing for {route} / {phase}")
    for field in EPOCH_FIELDS.values():
        points = record[field]["samples"]
        if len(points) != args.repeats or any(not math.isfinite(x) or x <= 0 for x in points):
            raise ValueError(f"Bad whole-epoch timing for {route}")


def run(args):
    sys.path.insert(0, str(args.kda_root))
    from kda_experiments.selection import confirm

    fixture = json.loads(args.manifest.read_text())
    if fixture["fixture_sha256"] != file_hash(args.fixture):
        raise ValueError("The dataset fixture differs from its manifest")
    binary_sha = file_hash(args.binary)
    if binary_sha != args.binary_sha256:
        raise ValueError("The benchmark executable differs from the frozen build")

    rounds = []
    gpu = None
    rng = random.Random(SEED)
    for _ in range(args.rounds):
        order = list(ROUTES)
        rng.shuffle(order)
        records = {}
        for route in order:
            command = [str(args.binary), str(args.fixture), route,
                       str(args.epochs), str(args.repeats)]
            result = subprocess.run(command, check=False, text=True,
                                    capture_output=True, timeout=20)
            if result.returncode:
                raise RuntimeError(f"{route} exited {result.returncode}: {result.stderr.strip()}")
            record = json.loads(result.stdout)
            verify(record, route, fixture, args)
            if gpu is None:
                gpu = record["gpu_uuid"]
            if record["gpu_uuid"] != gpu:
                raise ValueError("All paired runs must use the same GPU")
            records[route] = record
        rounds.append({"order": order, "records": records})

    def metrics(record):
        return {
            **record["diagnostic_ms_per_batch"],
            "setup_ms": record["setup_ms"],
            "dataset_upload_ms": record["dataset_upload_ms"],
            "layout_and_route_setup_ms": record["layout_and_route_setup_ms"],
            **{metric: record[field]["median"] for metric, field in EPOCH_FIELDS.items()},
            "live_bytes": record["live_bytes"],
        }

    def paired(baseline, candidate, metric):
        def points(route):
            return [{"dataset": [metrics(r["records"][route])[metric]]} for r in rounds]
        return confirm([{"id": "dataset"}], points(baseline), points(candidate),
                       "minimize", bootstrap_samples=4000, seed=SEED, confidence=0.95)

    summary = {
        route: {metric: statistics.median(metrics(r["records"][route])[metric] for r in rounds)
                for metric in metrics(rounds[0]["records"][route])}
        for route in ROUTES
    }
    report = {
        "binary_sha256": binary_sha,
        "driver_sha256": file_hash(Path(__file__)),
        "kda_selection_sha256": file_hash(args.kda_root / "kda_experiments/selection.py"),
        "fixture": {k: v for k, v in fixture.items() if k != "ordered_assay_columns"},
        "gpu_uuid": gpu,
        "epochs_per_sample": args.epochs,
        "repeats": args.repeats,
        "paired_rounds": rounds,
        "median_by_route": summary,
        "baseline_time_over_candidate_with_paired_95pct_interval": {
            candidate: {
                baseline: {metric: paired(baseline, candidate, metric)
                           for metric in EPOCH_FIELDS}
                for baseline in baselines
            }
            for candidate, baselines in (("tiled", ("csr", "cusparse")),
                                         ("tiled-fused", ("tiled", "csr", "cusparse")))
        },
    }
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"gpu_uuid": gpu, "median_by_route": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("binary", "fixture", "manifest", "kda-root", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if not (1 <= args.epochs <= 10 and 2 <= args.repeats <= 8 and 2 <= args.rounds <= 9):
        parser.error("Counts exceed the bounded benchmark envelope")
    run(args)
