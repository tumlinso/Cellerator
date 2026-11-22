#!/usr/bin/env python3
"""Paired evaluation of the native feature-major sparse training paths."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import subprocess
import sys

ROUTES = ("csr", "csr-transpose", "cusparse", "cusparse-alg2", "contiguous", "grouped")
TRACES = {
    "pbmc3k": ("pbmc3k-support-512.json", "e8ddfb1e439a32cb0843386b8d2e4c376aedba2ed19e3bd7b9aed72d98d45acd"),
    "gse147520": ("gse147520-support-256.json", "4f00eb8d42e59afde91120dc214af458e37d93405de3cb9650094acaf1d33aac"),
}
CASES = (*TRACES, "low-sharing", "skewed")
MATRIX_MARKET_ROWS = {"gse147520-4096": 4096, "gse147520-16384": 16384}
OPTIONAL_CASES = ("shared-tiles", *MATRIX_MARKET_ROWS)
HORIZONS = ("1", "8", "64")
SEED = 20261121


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fixture(name: str, traces: Path, path: Path, market_sources=None) -> dict:
    if name in MATRIX_MARKET_ROWS:
        source = market_sources[name]
        return market_fixture(name, source, path)
    if name in TRACES:
        filename, expected_sha = TRACES[name]
        source = traces / filename
        if sha256(source) != expected_sha:
            raise ValueError(f"Unexpected trace checksum: {source}")
        raw = json.loads(source.read_text(encoding="utf-8"))
        rows = raw["row_count"]
        features = raw["column_count"]
        offsets = raw["row_offsets"]
        columns = raw["column_indices"]
        metadata = {"trace_file_sha256": expected_sha, "trace_id": raw["trace_id"]}
    elif name == "shared-tiles":
        rows, features = 8193, 16384
        offsets, columns = [0], []
        for row in range(rows):
            first = ((row // 32) * 73 % (features // 32)) * 32
            columns.extend(range(first, first + 32))
            offsets.append(len(columns))
        metadata = {"synthetic": name}
    else:
        rows, features = 2049, 32768
        offsets, columns = [0], []
        for row in range(rows):
            if row % 31 != 30:
                row_columns = {(row * 197 + band * 4093) % features for band in range(16)}
                if name == "skewed":
                    row_columns |= set(range(64 if row % 16 == 0 else 8))
                columns.extend(sorted(row_columns))
            offsets.append(len(columns))
        metadata = {"synthetic": name}
    with path.open("wb") as out:
        out.write(struct.pack("<III", rows, features, len(columns)))
        for sequence in (offsets, columns):
            content = array.array("I", sequence)
            if content.itemsize != 4:
                raise RuntimeError("32-bit native unsigned integers are required")
            if sys.byteorder != "little":
                content.byteswap()
            content.tofile(out)
    return {**metadata, "rows": rows, "features": features, "nnz": len(columns), "sha256": sha256(path)}


def market_fixture(name: str, source: Path, path: Path) -> dict:
    import numpy as np
    from scipy.io import mmread

    original = mmread(source)
    rows, features = original.shape
    if rows != MATRIX_MARKET_ROWS[name] or features != 26587:
        raise ValueError(f"Unexpected all-column GSE147520 dimensions: {source}")
    csr = original.tocsr()
    if csr.nnz != original.nnz:
        raise ValueError(f"Matrix Market source has duplicate coordinates: {source}")
    del original
    sidecar = source.with_suffix(".rows.npy")
    row_ids = np.load(sidecar, allow_pickle=False)
    if row_ids.shape != (rows,) or np.unique(row_ids).size != rows:
        raise ValueError(f"Expected one distinct source row ID per row: {sidecar}")
    with path.open("wb") as out:
        out.write(struct.pack("<III", rows, features, csr.nnz))
        np.asarray(csr.indptr, dtype="<u4").tofile(out)
        np.asarray(csr.indices, dtype="<u4").tofile(out)
    return {
        "matrix_market_sha256": sha256(source),
        "source_rows_sha256": sha256(sidecar),
        "rows": rows,
        "features": features,
        "nnz": csr.nnz,
        "sha256": sha256(path),
    }


def run(args, path: Path, route: str, horizon: int) -> dict:
    command = [str(args.binary), str(path), route, args.mode, str(horizon),
               str(args.warmup), str(args.repeats)]
    res = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if res.returncode:
        raise RuntimeError(f"{route} failed for {path.stem}: {res.stderr.strip()}")
    result = json.loads(res.stdout)
    expected_uuid = args.expected_gpu.lower().removeprefix("gpu-").replace("-", "")
    if result["gpu_uuid_bytes"] != expected_uuid or result["compute_capability"] != 70:
        raise RuntimeError("Benchmark ran on a different GPU from the declared V100")
    if result["oracle"] != "pass" or set(result["horizons"]) != (set(HORIZONS) if horizon == 0 else {str(horizon)}):
        raise RuntimeError("Benchmark result is missing oracle evidence or required horizons")
    for samples in result["horizons"].values():
        if len(samples) != args.repeats:
            raise RuntimeError("Unexpected sample count")
        for sample in samples:
            if any(v <= 0 or not math.isfinite(v) for v in sample.values()):
                raise RuntimeError("Invalid timing sample")
    return result


def median_timing(rounds: list[dict], horizon: str, key: str) -> float:
    return statistics.median(statistics.median(
        sample[key] for sample in record["horizons"][horizon]) for record in rounds)


def evaluate_case(args, name: str, path: Path, confirm) -> tuple[dict, dict]:
    rng = random.Random(f"{SEED}-{name}-{args.mode}")
    screening = {route: [] for route in ROUTES}
    for _ in range(args.rounds):
        order = list(ROUTES)
        rng.shuffle(order)
        for route in order:
            screening[route].append(run(args, path, route, 0))
    timings = {h: {route: {
        "amortized_step_ms": median_timing(rounds, h, "amortized_step_ms"),
        "wall_step_ms": median_timing(rounds, h, "wall_step_ms"),
        "gpu_step_ms": median_timing(rounds, h, "gpu_step_ms"),
    } for route, rounds in screening.items()} for h in HORIZONS}

    def fastest(horizon: str, routes) -> str:
        return min(routes, key=lambda route: timings[horizon][route]["amortized_step_ms"])

    winner_tiles = fastest("64", ROUTES[-2:])
    winner_baseline = fastest("64", ROUTES[:-2])
    finals = {route: [] for route in (winner_baseline, winner_tiles)}
    for _ in range(args.confirmation_rounds):
        order = list(finals)
        rng.shuffle(order)
        for route in order:
            finals[route].append(run(args, path, route, 64))

    def samples(route: str) -> list[dict]:
        return [{name: [sample["amortized_step_ms"] for sample in record["horizons"]["64"]]}
                for record in finals[route]]

    paired = confirm([{"id": name}], samples(winner_baseline), samples(winner_tiles),
                     "minimize", bootstrap_samples=4000, seed=SEED, confidence=0.95)
    reduced = 1 - 1 / paired["score"]
    threshold_met = reduced >= 0.05 and paired["ci_low"] > 1.0
    minimum_screened_horizon = next((int(h) for h in HORIZONS
        if timings[h][winner_tiles]["amortized_step_ms"]
        < timings[h][fastest(h, ROUTES[:-2])]["amortized_step_ms"]), None)
    summary = {
        "timing_medians": timings,
        "median_setup_ms": {r: statistics.median(d["setup_ms"] for d in vals)
                            for r, vals in screening.items()},
        "median_sparse_update_ms": {r: statistics.median(d["sparse_update_ms"] for d in vals)
                                    for r, vals in screening.items()},
        "median_forward_ms": {r: statistics.median(d["forward_ms"] for d in vals)
                              for r, vals in screening.items()},
        "median_backward_ms": {r: statistics.median(d["backward_ms"] for d in vals)
                               for r, vals in screening.items()},
        "feature_records": {r: vals[0]["feature_records"]
                            for r, vals in screening.items() if vals[0]["feature_records"]},
        "live_device_array_bytes": {r: vals[0]["live_device_array_bytes"]
                                    for r, vals in screening.items()},
        "fastest_tiled_at_64": winner_tiles,
        "fastest_baseline_at_64": winner_baseline,
        "minimum_screened_horizon_with_gain": minimum_screened_horizon,
        "paired_confirmation_at_64": paired,
        "fractional_time_reduction_at_64": reduced,
        "threshold_met": threshold_met,
    }
    for phase in ("weight_gradient_ms", "sparse_gradient_ms"):
        if all(phase in record for rounds in screening.values() for record in rounds):
            summary[f"median_{phase}"] = {
                route: statistics.median(record[phase] for record in rounds)
                for route, rounds in screening.items()
            }
    return {"screening": screening, "confirmation": finals}, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--kda-root", type=Path, required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("learn", "refresh", "frozen"), required=True)
    parser.add_argument("--case", choices=(*CASES, *OPTIONAL_CASES), action="append")
    parser.add_argument("--matrix-market", type=Path, nargs=2, action="append", default=[],
                        metavar=("CASE", "PATH"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--confirmation-rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()
    if not (2 <= args.rounds <= 10 and 2 <= args.confirmation_rounds <= 15 and
            2 <= args.repeats <= 15 and 0 <= args.warmup <= 40):
        parser.error("Round, repeat, or warmup count is outside the bounded campaign envelope")
    args.binary = args.binary.resolve(strict=True)
    market_sources = {}
    for raw_case, source in args.matrix_market:
        name = raw_case.as_posix()
        if name not in MATRIX_MARKET_ROWS or name in market_sources:
            parser.error(f"Unknown or duplicate Matrix Market case: {name}")
        market_sources[name] = source.resolve(strict=True)
    if (set(args.case or CASES) & set(MATRIX_MARKET_ROWS)) - market_sources.keys():
        parser.error("Each selected Matrix Market case requires --matrix-market CASE PATH")
    uuid = args.expected_gpu.lower().removeprefix("gpu-").replace("-", "")
    if len(uuid) != 32 or any(x not in "0123456789abcdef" for x in uuid):
        parser.error("Expected a physical GPU UUID")
    sys.path.insert(0, str(args.kda_root.resolve(strict=True)))
    from kda_experiments.selection import confirm

    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_config = {
        "binary_sha256": sha256(args.binary),
        "driver_sha256": sha256(Path(__file__)),
        "selection_sha256": sha256(args.kda_root / "kda_experiments" / "selection.py"),
        "expected_gpu": args.expected_gpu,
        "architecture": "sm_70",
        "mode": args.mode,
        "screening_rounds": args.rounds,
        "confirmation_rounds": args.confirmation_rounds,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "horizons": [int(h) for h in HORIZONS],
        "seed": SEED,
        "cases": {},
    }
    raw, summary, lines = {}, {}, []
    for case in args.case or CASES:
        path = args.output_dir / f"{case}.csr"
        run_config["cases"][case] = fixture(case, args.traces, path, market_sources)
        write_json(args.output_dir / "run_config.json", run_config)
        raw[case], summary[case] = evaluate_case(args, case, path, confirm)
        write_json(args.output_dir / "results.json", raw)
        write_json(args.output_dir / "summary.json", summary)
        cell = summary[case]
        paired = cell["paired_confirmation_at_64"]
        lines.append(
            f"{args.mode} {case} at 64: {cell['fastest_tiled_at_64']} vs "
            f"{cell['fastest_baseline_at_64']}: {1.0 / paired['score']:.3f}x baseline time, "
            f"speedup 95% CI [{paired['ci_low']:.4f}, {paired['ci_high']:.4f}], "
            f"threshold met: {cell['threshold_met']}"
        )
        (args.output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(lines[-1], flush=True)


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


if __name__ == "__main__":
    main()
