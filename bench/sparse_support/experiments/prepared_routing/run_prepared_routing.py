#!/usr/bin/env python3
"""Admitted-device correctness and complete-route-path comparison.

This script is intentionally not run by ordinary tests: it performs CUDA
allocations, launches, and synchronizations. Root must admit the finite device
job before invoking it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]
                      / "python" / "cellerator" / "torch"))
from prepared_routing import (
    PreparedEntityMechanismRouting,
    PreparedRoutingIdentity,
)


SHAPES = (
    ("ranked_32x8", 32, 8),
    ("event_32x12", 32, 12),
    ("stress_256x64", 256, 64),
    ("stress_1024x128", 1024, 128),
)
HORIZONS = (1, 12, 16, 64)
ENTITY_COUNT = 4096
MECHANISM_COUNT = 16


def route_rows() -> tuple[tuple[int, ...], ...]:
    rows = []
    for entity in range(ENTITY_COUNT):
        if entity % 17 == 0:
            rows.append(())
            continue
        first = (entity * 7 + 3) % MECHANISM_COUNT
        second = (entity * 13 + 5) % MECHANISM_COUNT
        routes = {first}
        if entity % 3 != 0:
            routes.add(second)
        rows.append(tuple(sorted(routes)))
    return tuple(rows)


def oracle(rows: tuple[tuple[int, ...], ...], entity_ids) -> tuple[list[int], list[int]]:
    offsets = [0]
    mechanisms: list[int] = []
    for row in entity_ids:
        active = set()
        for entity in row:
            if entity < 0:
                continue
            if entity >= len(rows):
                raise ValueError("mechanism-route entity is outside the prepared domain")
            active.update(rows[entity])
        mechanisms.extend(sorted(active))
        offsets.append(len(mechanisms))
    return offsets, mechanisms


def incumbent_python(ids: torch.Tensor, rows) -> tuple[torch.Tensor, torch.Tensor]:
    """Current affected_mechanism_workset path, including both transfers."""
    output_rows: list[list[int]] = []
    for entity_row in ids.detach().cpu().tolist():
        active = set()
        for entity in entity_row:
            if entity >= 0:
                if entity >= len(rows):
                    raise ValueError("mechanism-route entity is outside the prepared domain")
                active.update(rows[entity])
        output_rows.append(sorted(active))
    offsets = [0]
    mechanisms: list[int] = []
    for row in output_rows:
        mechanisms.extend(row)
        offsets.append(len(mechanisms))
    return (torch.tensor(offsets, dtype=torch.int64, device=ids.device),
            torch.tensor(mechanisms, dtype=torch.int32, device=ids.device))


def make_dense_route_matrix(rows, device: torch.device) -> torch.Tensor:
    dense = torch.zeros((len(rows), MECHANISM_COUNT), dtype=torch.bool)
    entities: list[int] = []
    mechanisms: list[int] = []
    for entity, row in enumerate(rows):
        for mechanism in row:
            entities.append(entity)
            mechanisms.append(mechanism)
    if entities:
        dense[torch.tensor(entities), torch.tensor(mechanisms)] = True
    return dense.to(device=device)


def torch_dense_route_union(ids: torch.Tensor,
                            dense_routes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    visible = ids >= 0
    route_rows = dense_routes[ids.clamp_min(0).to(torch.int64)]
    selected = (route_rows & visible.unsqueeze(-1)).any(dim=1)
    counts = selected.sum(dim=1, dtype=torch.int64)
    offsets = torch.cat((counts.new_zeros((1,)), counts.cumsum(0)))
    mechanisms = selected.nonzero(as_tuple=False)[:, 1].to(torch.int32)
    return offsets, mechanisms


def make_inputs(cells: int, participants: int, seed: int,
                device: torch.device, count: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = []
    for sample in range(count):
        ids = torch.randint(0, ENTITY_COUNT, (cells, participants),
                            generator=generator, dtype=torch.int64)
        if participants:
            padding = torch.rand((cells, participants), generator=generator) < 0.1
            ids[padding] = -1
            if participants >= 3:
                ids[0, 1] = ids[0, 0]  # duplicate term
                ids[0, 2] = 0          # fixed low ordinal; order stays arbitrary
        inputs.append(ids.to(device=device))
    return inputs


def materialize(result) -> tuple[list[int], list[int]]:
    offsets, mechanisms = result
    return offsets.cpu().tolist(), mechanisms.cpu().tolist()


def input_digest(ids: torch.Tensor) -> str:
    host = ids.detach().cpu().contiguous()
    return hashlib.sha256(host.numpy().tobytes()).hexdigest()


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def timed_call(function, ids: torch.Tensor, device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter()
    result = function(ids)
    synchronize(device)
    elapsed = time.perf_counter() - start
    del result
    return elapsed


def timed_prepare(function, device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter()
    value = function()
    synchronize(device)
    elapsed = time.perf_counter() - start
    del value
    synchronize(device)
    return elapsed


def timed_cold_first_query(owner_factory, query, ids: torch.Tensor,
                           device: torch.device) -> float:
    """Time a fresh owner through its first complete query and completion.

    This includes owner construction, device upload, first-query workspace
    reservation, output construction, and all synchronization needed to make
    the result usable. PyTorch, the extension, CUDA context, and route rows are
    initialized before this measurement; it is not process-cold timing.
    """
    synchronize(device)
    start = time.perf_counter()
    owner = owner_factory()
    result = query(owner, ids)
    synchronize(device)
    elapsed = time.perf_counter() - start
    del result
    del owner
    synchronize(device)
    return elapsed


def query_prepared(router, ids, identity):
    return router.query(ids, identity=identity)


def correctness_checks(native, rows, device, identity, router, input_samples):
    dense_routes = make_dense_route_matrix(rows, device)
    for ids in input_samples:
        expected = oracle(rows, ids.cpu().tolist())
        for actual in (incumbent_python(ids, rows),
                       torch_dense_route_union(ids, dense_routes),
                       query_prepared(router, ids, identity)):
            if materialize(actual) != expected:
                raise AssertionError("route union differs from the independent Python oracle")

    empty = torch.empty((3, 0), dtype=torch.int64, device=device)
    if materialize(query_prepared(router, empty, identity)) != ([0, 0, 0, 0], []):
        raise AssertionError("empty participant rows did not produce empty unions")
    all_padding = torch.full((3, 5), -1, dtype=torch.int64, device=device)
    if materialize(query_prepared(router, all_padding, identity)) != ([0, 0, 0, 0], []):
        raise AssertionError("all-padding input did not produce empty unions")
    invalid = torch.tensor([[ENTITY_COUNT]], dtype=torch.int64, device=device)
    try:
        query_prepared(router, invalid, identity)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-domain nonnegative ID did not fail closed")


def median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def paired_improvement(base: list[float], candidate: list[float], seed: int):
    if len(base) != len(candidate) or not base:
        raise ValueError("paired performance samples have mismatched lengths")
    samples = [(left - right) / left for left, right in zip(base, candidate)
               if left > 0.0]
    if len(samples) != len(base):
        raise ValueError("paired performance sample contains a nonpositive baseline")
    rng = random.Random(seed)
    bootstrap = sorted(statistics.median(
        [samples[rng.randrange(len(samples))] for _ in samples])
        for _ in range(5000))
    low = bootstrap[int(0.025 * (len(bootstrap) - 1))]
    high = bootstrap[int(0.975 * (len(bootstrap) - 1))]
    return {
        "median_percent": 100.0 * statistics.median(samples),
        "bootstrap_95_percent_interval": [100.0 * low, 100.0 * high],
    }


def load_native_module(path: Path):
    spec = importlib.util.spec_from_file_location("_torch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Torch extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(native, rows, name, cells, participants, device, seed,
             warmups, repeats):
    identity = PreparedRoutingIdentity(
        owner_identity=1, relation_id=2, structure_id=3,
        structure_epoch=4, route_generation=5, binding_generation=6)
    inputs = make_inputs(cells, participants, seed, device, max(repeats, warmups, 1))
    sparse_factory = lambda: PreparedEntityMechanismRouting(
        native, rows, MECHANISM_COUNT, identity, device.index)
    dense_factory = lambda: make_dense_route_matrix(rows, device)

    # Initialize the context outside all measurements, then prove output identity.
    sparse = sparse_factory()
    dense = dense_factory()
    correctness_checks(native, rows, device, identity, sparse, inputs)
    input_digests = [input_digest(ids) for ids in inputs]

    # Keep constructor-only setup as a diagnostic. Lifecycle accounting below
    # instead uses a fresh owner through its first complete query.
    constructor_sparse = [timed_prepare(sparse_factory, device)
                          for _ in range(repeats)]
    constructor_dense = [timed_prepare(dense_factory, device)
                         for _ in range(repeats)]

    warm_sparse = lambda ids: query_prepared(sparse, ids, identity)
    cold_queries = {
        "torch_dense": lambda ids: timed_cold_first_query(
            dense_factory,
            lambda owner, values: torch_dense_route_union(values, owner),
            ids, device),
        "prepared_postings": lambda ids: timed_cold_first_query(
            sparse_factory,
            lambda owner, values: query_prepared(owner, values, identity),
            ids, device),
    }
    functions = {
        "incumbent_python": lambda ids: incumbent_python(ids, rows),
        "torch_dense": lambda ids: torch_dense_route_union(ids, dense),
        "prepared_postings": warm_sparse,
    }
    for _ in range(warmups):
        for fn in functions.values():
            fn(inputs[0])
            synchronize(device)

    paired: dict[str, list[float]] = {
        label: [0.0] * repeats for label in functions}
    cold_first: dict[str, list[float]] = {
        label: [0.0] * repeats for label in cold_queries}
    repeat_records = {}
    order = list(range(repeats))
    random.Random(seed + 97).shuffle(order)
    timing_order = random.Random(seed + 193)
    for sample in order:
        ids = inputs[sample]
        labels = ["cold:" + label for label in cold_queries]
        labels.extend("warm:" + label for label in functions)
        timing_order.shuffle(labels)
        warm_times = {}
        cold_times = {}
        for label in labels:
            phase, method = label.split(":", maxsplit=1)
            if phase == "cold":
                elapsed = cold_queries[method](ids)
                cold_first[method][sample] = elapsed
                cold_times[method] = elapsed
            else:
                elapsed = timed_call(functions[method], ids, device)
                paired[method][sample] = elapsed
                warm_times[method] = elapsed
        repeat_records[sample] = {
            "repeat_id": f"{name}-{sample:03d}",
            "input_sample_id": sample,
            "input_sha256": input_digests[sample],
            "measurement_order": labels,
            "cold_first_query_ms": {
                method: elapsed * 1000.0
                for method, elapsed in cold_times.items()},
            "warm_query_ms": {
                method: elapsed * 1000.0
                for method, elapsed in warm_times.items()},
        }

    query_ms = {label: median_ms(values) for label, values in paired.items()}
    cold_first_ms = {
        "incumbent_python": 0.0,
        "torch_dense": median_ms(cold_first["torch_dense"]),
        "prepared_postings": median_ms(cold_first["prepared_postings"]),
    }
    constructor_setup_ms = {
        "incumbent_python": 0.0,
        "torch_dense": median_ms(constructor_dense),
        "prepared_postings": median_ms(constructor_sparse),
    }
    lifecycle = {}
    paired_comparison = {}
    for horizon in HORIZONS:
        lifecycle[horizon] = {}
        lifecycle_samples = {
            "incumbent_python": [
                horizon * sample for sample in paired["incumbent_python"]],
            "torch_dense": [cold_first["torch_dense"][i]
                            + (horizon - 1) * paired["torch_dense"][i]
                            for i in range(repeats)],
            "prepared_postings": [cold_first["prepared_postings"][i]
                                  + (horizon - 1)
                                  * paired["prepared_postings"][i]
                                  for i in range(repeats)],
        }
        for label in functions:
            total = median_ms(lifecycle_samples[label])
            lifecycle[horizon][label] = {
                "total_ms": total,
                "per_step_ms": total / horizon,
            }
        paired_comparison[horizon] = {}
        for candidate in ("torch_dense", "prepared_postings"):
            paired_comparison[horizon][candidate] = {
                baseline: paired_improvement(values, lifecycle_samples[candidate],
                    seed + horizon * 31 + index)
                for index, (baseline, values) in enumerate(lifecycle_samples.items())
                if baseline != candidate
            }
        for sample, record in repeat_records.items():
            record.setdefault("lifecycle_ms", {})[str(horizon)] = {
                label: lifecycle_samples[label][sample] * 1000.0
                for label in functions}
    return {
        "case": name,
        "cells": cells,
        "participants": participants,
        "entity_count": ENTITY_COUNT,
        "mechanism_count": MECHANISM_COUNT,
        "reuse_horizons": list(HORIZONS),
        "constructor_setup_ms_median": constructor_setup_ms,
        "cold_first_query_ms_median": cold_first_ms,
        "complete_route_query_ms_median": query_ms,
        "lifecycle": lifecycle,
        "paired_lifecycle_improvement": paired_comparison,
        "paired_repeat_records": [repeat_records[i] for i in range(repeats)],
        "samples_per_query": repeats,
        "warmups_per_query": warmups,
        "timing_note": (
            "Each complete query is wall-clock timed through CUDA completion. "
            "Prepared query includes term adaptation, reverse-posting kernels, "
            "the explicit status/count synchronization, and exact-ID emission. "
            "Lifecycle total is fresh-owner construction plus its first "
            "complete query, then horizon-1 warm complete queries. The first "
            "prepared query includes workspace reservation; the first dense "
            "query includes dense matrix construction and upload. Incumbent "
            "lifecycle is horizon complete queries. PyTorch, the extension, "
            "CUDA context, route rows, and correctness checks are already "
            "initialized; cold is owner-cold, not process-cold. No latent "
            "transition compute is included."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--torch-module-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=14491)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats must be positive")

    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    native = load_native_module(args.torch_module_file)
    rows = route_rows()
    results = []
    for ordinal, (name, cells, participants) in enumerate(SHAPES):
        results.append(run_case(native, rows, name, cells, participants,
                                device, args.seed + ordinal,
                                args.warmups, args.repeats))
    report = {
        "seed": args.seed,
        "device_index": args.device_index,
        "torch_version": torch.__version__,
        "initialization_scope": (
            "PyTorch, extension, CUDA context, route rows, and correctness "
            "checks are initialized before timing; cold_first_query means a "
            "fresh route owner through its first complete query, not a "
            "process-cold measurement."),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
