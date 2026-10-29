#!/usr/bin/env python3
"""Run the existing Torch sparse-training oracle against one CE-GEO layout."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DRIVER = REPO / "bench/sparse_support/packed_training_benchmark.py"
FIXTURE_ROOT = Path("/tmp/cellerator_scaled_sparse_fixtures_887437fc")
PACKAGE_ROOT = Path(os.environ.get("CELLERATOR_SPARSE_PACKAGE_ROOT", ""))


def import_driver():
    spec = importlib.util.spec_from_file_location("packed_training_benchmark", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark driver: {DRIVER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def support_for_case(bench, SupportSpec, case_name: str):
    if case_name == "event_local":
        bench.ENTITIES = 32
        return bench.make_support(SupportSpec), 32, None, None, None
    if case_name == "small":
        bench.ENTITIES = 64
        bundle = FIXTURE_ROOT / "small"
        arrays, manifest, manifest_sha = bench.load_raw_bundle(bundle, (
            "edge_ids", "source_entity_id", "participant_entity_id",
        ))
        metadata = manifest["metadata"]
        support = SupportSpec(
            tuple(map(int, arrays["edge_ids"])),
            tuple(map(int, arrays["source_entity_id"])),
            tuple(map(int, arrays["participant_entity_id"])), 1,
        )
        return support, int(metadata["entities"]), None, manifest, manifest_sha
    if case_name == "large_ranked":
        bench.ENTITIES = 32
        bundle = FIXTURE_ROOT / "large-ranked"
        arrays, manifest, manifest_sha = bench.load_raw_bundle(bundle, (
            "edge_ids", "source_entity_id", "participant_entity_id",
            "source_edge_indptr", "source_edge_ordinals", "ranked_entity_id",
            "ranked_position", "ranked_presence_offsets",
            "ranked_presence_entity_ids", "ranked_presence_values",
            "ranked_presence_view_id",
        ))
        metadata = manifest["metadata"]
        support = SupportSpec(
            tuple(map(int, arrays["edge_ids"])),
            tuple(map(int, arrays["source_entity_id"])),
            tuple(map(int, arrays["participant_entity_id"])), 1,
        )
        return support, int(metadata["entities"]), arrays, manifest, manifest_sha
    raise ValueError(f"unsupported geometry case: {case_name}")


def image_for(candidate: str, case_name: str):
    image_name = "canonical" if candidate == "simt" else candidate
    case_dir = {"small": "small", "large_ranked": "large-ranked",
                "event_local": "event_local"}[case_name]
    metadata = json.loads((HERE / "evidence/prepared_images/prepared_images.json").read_text())
    row = metadata["cases"][case_dir]["candidates"][image_name]
    image_path = HERE / "evidence/prepared_images" / case_dir / f"{image_name}.cegeo"
    compiled = {
        "image": image_path.read_bytes(),
        "host_compile_ms": float(row["host_compile_ms"]),
        "source_group_members": row["source_group_members"],
        "destination_group_members": row["destination_group_members"],
        "geometry_edge_ids_in_support_order": row["geometry_edge_ids_in_support_order"],
        "projection_id": int(row["projection_id"]),
        "tile_count": int(row["tile_count"]),
        "residual_region_count": int(row["residual_region_count"]),
        "residual_edge_count": int(row["residual_edge_count"]),
        "mma_edge_count": int(row["mma_edge_count"]),
        "image_path": str(image_path),
        "image_sha256": row["image_sha256"],
    }
    return compiled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("small", "large_ranked", "event_local"),
                        required=True)
    parser.add_argument("--candidate", choices=("simt", "canonical", "reordered",
                                                   "hybrid8", "hybrid16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--setup-pairs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        parser.error("geometry Torch benchmark requires the admitted Device Scheduler job")
    if not PACKAGE_ROOT.is_dir():
        parser.error("CELLERATOR_SPARSE_PACKAGE_ROOT must name the qualified Torch package")
    if min(args.warmup, args.rounds, args.iterations, args.setup_pairs) <= 0:
        parser.error("warmup, rounds, iterations, and setup pairs must be positive")

    sys.path.insert(0, str(REPO / "bench/sparse_support"))
    sys.path.insert(0, str(PACKAGE_ROOT))
    import torch
    from cellerator.torch.sparse_support import SupportSpec

    bench = import_driver()
    package = importlib.import_module("cellerator.torch.sparse_support")
    if not Path(package.__file__).resolve().is_relative_to(PACKAGE_ROOT.resolve()):
        raise RuntimeError(f"imported Torch package is outside the frozen root: {package.__file__}")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    init_start = time.perf_counter()
    torch.cuda.init()
    device = torch.cuda.current_device()
    torch.empty((1,), device=device).zero_()
    torch.cuda.synchronize(device)
    common_init_ms = (time.perf_counter() - init_start) * 1000.0

    support, entities, fixture_arrays, manifest, manifest_sha = support_for_case(
        bench, SupportSpec, args.case)
    compiled = image_for(args.candidate, args.case)
    case_name = {"small": "ranked_batch", "large_ranked": "large_ranked",
                 "event_local": "event_local"}[args.case]
    benchmark_args = SimpleNamespace(
        fixture_arrays=fixture_arrays,
        fixture_manifest=manifest,
        fixture_bundle=(str(FIXTURE_ROOT / "large-ranked")
                        if args.case == "large_ranked" else None),
        fixture_manifest_sha256=manifest_sha,
        setup_pairs=args.setup_pairs,
        warmup=args.warmup,
        rounds=args.rounds,
        iterations=args.iterations,
        common_cuda_init_ms=common_init_ms,
    )
    result = bench.benchmark_case(
        torch, package, support, case_name, args.seed, benchmark_args, compiled)
    result["case"]["geometry_probe_case"] = args.case
    result["case"]["geometry_candidate"] = args.candidate
    result["case"]["geometry_image"] = {
        key: compiled[key] for key in (
            "image_path", "image_sha256", "host_compile_ms", "tile_count",
            "mma_edge_count", "residual_edge_count", "residual_region_count",
        )
    }
    result["case"]["support_edge_count"] = len(support.edge_ids)
    result["case"]["logical_entities"] = entities
    result["fixture_manifest_sha256"] = manifest_sha
    result["device"] = {
        "index": device,
        "name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "case": args.case, "candidate": args.candidate,
        "output": str(args.output), "checks": result["checks"],
        "training_step_median_ms": {
            path: result["phases"]["training"][path]["synchronized_wall_ms_per_step"]["median"]
            for path in ("simt", "packed")
        },
        "geometry": result["case"]["geometry_image"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
