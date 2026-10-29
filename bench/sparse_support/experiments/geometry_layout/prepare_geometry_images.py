#!/usr/bin/env python3
"""Compile isolated canonical, reordered, and thresholded CE-GEO probe images."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "bench/sparse_support"))
HOST_PROBE = HERE / "host_geometry_probe.py"
spec = importlib.util.spec_from_file_location("host_geometry_probe", HOST_PROBE)
host_probe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(host_probe)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case(fixture_root: Path, name: str, SupportSpec, bench):
    if name == "event_local":
        support = bench.make_support(SupportSpec)
        return {
            "name": name, "support": support, "entities": bench.ENTITIES,
            "manifest_sha256": None, "arrays": None,
        }
    arrays, manifest, manifest_sha = host_probe.load_bundle(
        fixture_root / name,
        ("edge_ids", "source_entity_id", "participant_entity_id",
         "edge_module_id", "ranked_entity_id", "ranked_position",
         "source_edge_indptr", "source_edge_ordinals"),
    )
    metadata = manifest["metadata"]
    support = SupportSpec(
        tuple(map(int, arrays["edge_ids"])),
        tuple(map(int, arrays["source_entity_id"])),
        tuple(map(int, arrays["participant_entity_id"])), 1,
    )
    return {"name": name, "support": support,
            "entities": int(metadata["entities"]),
            "manifest": manifest, "manifest_sha256": manifest_sha,
            "arrays": arrays}


def write_endpoints(case: dict, inverse: np.ndarray, directory: Path):
    source = np.asarray(case["support"].sources, dtype=np.uint32)
    destination = np.asarray(case["support"].destinations, dtype=np.uint32)
    physical_source = np.ascontiguousarray(inverse[source], dtype=np.uint32)
    physical_destination = np.ascontiguousarray(inverse[destination], dtype=np.uint32)
    source_path = directory / "source.bin"
    destination_path = directory / "destination.bin"
    source_path.write_bytes(physical_source.tobytes())
    destination_path.write_bytes(physical_destination.tobytes())
    return physical_source, physical_destination, source_path, destination_path


def _header_type():
    import ctypes

    class StructureIdentity(ctypes.Structure):
        _fields_ = [("schema_version", ctypes.c_uint32),
                    ("identity_version", ctypes.c_uint32),
                    ("value", ctypes.c_uint64)]

    class FeatureOrderIdentity(ctypes.Structure):
        _fields_ = [("schema_version", ctypes.c_uint32),
                    ("kind", ctypes.c_uint32),
                    ("feature_count", ctypes.c_uint32),
                    ("feature_axis_identity_version", ctypes.c_uint32),
                    ("feature_axis_identity", ctypes.c_uint64),
                    ("packing_geometry_identity", ctypes.c_uint64)]

    class Header(ctypes.Structure):
        _fields_ = [
            ("schema_version", ctypes.c_uint32), ("header_bytes", ctypes.c_uint32),
            ("image_bytes", ctypes.c_uint64), ("structure_identity", StructureIdentity),
            ("source_order", FeatureOrderIdentity),
            ("destination_order", FeatureOrderIdentity),
            ("provider_identity_low", ctypes.c_uint64), ("provider_identity_high", ctypes.c_uint64),
            ("capability_identity_low", ctypes.c_uint64), ("capability_identity_high", ctypes.c_uint64),
            ("logical_edge_count", ctypes.c_uint64), ("dense_width", ctypes.c_uint32),
            ("logical_edge_id_width", ctypes.c_uint8), ("reserved0", ctypes.c_uint8 * 3),
            ("source_group_count", ctypes.c_uint32), ("destination_group_count", ctypes.c_uint32),
            ("tile_count", ctypes.c_uint32), ("compact_slot_count", ctypes.c_uint32),
            ("residual_region_count", ctypes.c_uint32), ("schedule_entry_count", ctypes.c_uint32),
            ("value_map_count", ctypes.c_uint64), ("source_group_offset", ctypes.c_uint64),
            ("destination_group_offset", ctypes.c_uint64), ("tile_offset", ctypes.c_uint64),
            ("compact_slot_offset", ctypes.c_uint64), ("residual_region_offset", ctypes.c_uint64),
            ("residual_row_offset_offset", ctypes.c_uint64),
            ("residual_column_index_offset", ctypes.c_uint64),
            ("schedule_entry_offset", ctypes.c_uint64), ("value_map_offset", ctypes.c_uint64),
            ("reserved", ctypes.c_uint32 * 8),
        ]
    return Header


def image_metadata(image: bytes) -> dict:
    import ctypes
    header = _header_type().from_buffer_copy(image)
    return {
        "image_bytes": int(header.image_bytes),
        "tile_count": int(header.tile_count),
        "mma_edge_count": int(header.compact_slot_count),
        "residual_edge_count": int(header.logical_edge_count - header.compact_slot_count),
        "residual_region_count": int(header.residual_region_count),
        "value_map_count": int(header.value_map_count),
    }


def compile_hybrid(helper: Path, case: dict, inverse: np.ndarray,
                   order: np.ndarray, threshold: int, output_path: Path,
                   projection_id: int):
    work = output_path.parent / f"{output_path.stem}.inputs"
    work.mkdir(parents=True, exist_ok=True)
    _, _, source_path, destination_path = write_endpoints(case, inverse, work)
    start = time.perf_counter()
    proc = subprocess.run([
        str(helper), str(case["entities"]), str(len(case["support"].edge_ids)),
        str(threshold), str(source_path), str(destination_path), str(output_path),
    ], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"hybrid image helper failed: {proc.stderr}")
    image = output_path.read_bytes()
    metadata = json.loads(proc.stdout)
    metadata.update({
        "image": image,
        "host_compile_ms": (time.perf_counter() - start) * 1000.0,
        "source_group_members": order.tolist(),
        "destination_group_members": order.tolist(),
        "stable_support_edge_ids": list(case["support"].edge_ids),
        "geometry_edge_ids_in_support_order": list(range(len(case["support"].edge_ids))),
        "projection_id": projection_id,
        "structure_epoch": case["support"].structure_epoch,
        "image_sha256": sha256_bytes(image),
    })
    return metadata, proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path,
                        default=Path("/tmp/cellerator_scaled_sparse_fixtures_887437fc"))
    parser.add_argument("--helper", type=Path,
                        default=HERE / "evidence/geometry_image_probe")
    parser.add_argument("--case", choices=("all", "small", "large-ranked", "event_local"),
                        default="all")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "evidence/prepared_images")
    args = parser.parse_args()
    os.environ["PYTHONPATH"] = str(args.package_root.resolve()) + os.pathsep + \
        os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, str(args.package_root.resolve()))
    import cellerator.torch.sparse_support as package
    import packed_training_benchmark as bench

    if not args.helper.is_file():
        raise FileNotFoundError(args.helper)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "cellerator.geometry_layout_prepared_images.v1",
        "tool_identity": {
            "preparation_script_sha256": sha256_file(Path(__file__)),
            "host_probe_script_sha256": sha256_file(HOST_PROBE),
            "hybrid_builder_source_sha256": sha256_file(HERE / "hybrid_builder_probe.cc"),
            "geometry_image_probe_source_sha256": sha256_file(HERE / "geometry_image_probe.cc"),
            "hybrid_builder_binary_sha256": sha256_file(args.helper),
            "frozen_torch_identity_sha256": sha256_file(args.package_root.parent / "identity.json"),
            "frozen_torch_extension_sha256": sha256_file(
                args.package_root / "cellerator/torch/_torch.cpython-313-x86_64-linux-gnu.so"),
        },
        "cases": {},
    }
    case_names = (("small", "large-ranked", "event_local") if args.case == "all"
                  else (args.case,))
    for case_name in case_names:
        print(f"preparing {case_name}", flush=True)
        case = load_case(args.fixture_root, case_name, package.SupportSpec, bench)
        entities = case["entities"]
        source = np.asarray(case["support"].sources, dtype=np.int64)
        destination = np.asarray(case["support"].destinations, dtype=np.int64)
        identity = np.arange(entities, dtype=np.int64)
        if case_name == "event_local":
            layout_start = time.perf_counter()
            order = host_probe.neighbor_greedy_order(source, destination, entities)
            fixture_arrays = None
        else:
            fixture_arrays = case["arrays"]
            layout_start = time.perf_counter()
            order = host_probe.neighbor_greedy_order(source, destination, entities)
        layout_ms = (time.perf_counter() - layout_start) * 1000.0
        inverse = np.empty(entities, dtype=np.int64)
        inverse[order] = np.arange(entities, dtype=np.int64)
        case_dir = args.output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        candidates = {}
        identity_inverse = np.arange(entities, dtype=np.int64)
        canonical, _ = compile_hybrid(
            args.helper, case, identity_inverse, identity, 1,
            case_dir / "canonical.cegeo", 101)
        canonical_meta = {key: value for key, value in canonical.items()
                          if key != "image"}
        candidates["canonical"] = canonical_meta
        canonical_compile_ms = float(canonical["host_compile_ms"])
        print(f"  canonical: {canonical_meta['tile_count']} tiles, "
              f"{canonical_meta['image_bytes']} bytes, "
              f"{canonical_compile_ms:.2f} ms", flush=True)

        reordered, _ = compile_hybrid(
            args.helper, case, inverse, order, 1,
            case_dir / "reordered.cegeo", 102)
        reordered_meta = {key: value for key, value in reordered.items()
                          if key != "image"}
        candidates["reordered"] = reordered_meta
        compile_ms = float(reordered["host_compile_ms"])
        print(f"  reordered: {reordered_meta['tile_count']} tiles, "
              f"{reordered_meta['image_bytes']} bytes, "
              f"{compile_ms:.2f} ms", flush=True)

        thresholds = (8, 16)
        helper_logs = {}
        for index, threshold in enumerate(thresholds):
            label = f"hybrid{threshold}"
            metadata, helper_log = compile_hybrid(
                args.helper, case, inverse, order, threshold,
                case_dir / f"{label}.cegeo", 103 + index)
            candidates[label] = {key: value for key, value in metadata.items()
                                 if key != "image"}
            helper_logs[label] = helper_log
            print(f"  {label}: {metadata['tile_count']} tiles, "
                  f"{metadata['image_bytes']} bytes, "
                  f"{metadata['host_compile_ms']:.2f} ms", flush=True)
        summary["cases"][case_name] = {
            "entities": entities,
            "edge_count": len(case["support"].edge_ids),
            "fixture_manifest_sha256": case["manifest_sha256"],
            "graph_order_sha256": sha256_bytes(order.astype("<i4").tobytes()),
            "graph_order_prepare_ms_host": layout_ms,
            "candidates": candidates,
            "helper_logs": helper_logs,
            "image_paths": {
                label: f"{case_name}/{label}.cegeo"
                for label in candidates
            },
        }
    out_path = args.output_dir / "prepared_images.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {
        label: {key: value[key] for key in (
            "host_compile_ms", "tile_count", "mma_edge_count",
            "residual_edge_count", "residual_region_count", "value_map_count",
            "image_bytes", "image_sha256") if key in value}
        for label, value in row["candidates"].items()
    } for name, row in summary["cases"].items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
