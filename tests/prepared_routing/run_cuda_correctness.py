#!/usr/bin/env python3
"""GPU correctness job; run only after scheduler admission."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                      / "python" / "cellerator" / "torch"))
from prepared_routing import (  # noqa: E402
    PreparedEntityMechanismRouting,
    PreparedRoutingIdentity,
)


def oracle(routes, inputs):
    offsets = [0]
    values = []
    for row in inputs:
        active = set()
        for entity in row:
            if entity < 0:
                continue
            if entity >= len(routes):
                raise ValueError("entity outside domain")
            active.update(routes[entity])
        values.extend(sorted(active))
        offsets.append(len(values))
    return offsets, values


def check(router, identity, routes, ids):
    expected = oracle(routes, ids.cpu().tolist())
    actual_offsets, actual_values = router.query(ids, identity=identity)
    actual = actual_offsets.cpu().tolist(), actual_values.cpu().tolist()
    if actual != expected:
        raise AssertionError(f"mismatch: expected {expected}, got {actual}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--torch-module-file", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    spec = importlib.util.spec_from_file_location("_torch", args.torch_module_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Torch extension from {args.torch_module_file}")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    entity_count, mechanism_count = 4096, 16
    routes = []
    for entity in range(entity_count):
        if entity % 17 == 0:
            routes.append(())
            continue
        first = (entity * 7 + 3) % mechanism_count
        second = (entity * 13 + 5) % mechanism_count
        row = [second, first, first]
        if entity % 3 == 0:
            row = [first, first]
        routes.append(tuple(row))
    routes = tuple(routes)
    identity = PreparedRoutingIdentity(101, 102, 103, 104, 105, 106)
    router = PreparedEntityMechanismRouting(
        native, routes, mechanism_count, identity, args.device_index)

    fixed = torch.tensor(
        [[4, 2, 4, -1, 1], [3, 3, 0, 2, -1], [-1, -1, -1, -1, -1]],
        dtype=torch.int64, device=device)
    check(router, identity, routes, fixed)
    check(router, identity, routes, fixed.to(torch.int32))
    check(router, identity, routes,
          torch.empty((4, 0), dtype=torch.int64, device=device))
    for cells, participants in ((32, 8), (32, 12), (256, 64), (1024, 128)):
        generator = torch.Generator(device="cpu").manual_seed(cells * 1000 + participants)
        ids = torch.randint(0, entity_count, (cells, participants),
                            dtype=torch.int64, generator=generator)
        ids[:, -1] = -1
        if participants > 1:
            ids[:, 1] = ids[:, 0]
        check(router, identity, routes, ids.to(device))

    invalid = torch.tensor([[entity_count]], dtype=torch.int64, device=device)
    try:
        router.query(invalid, identity=identity)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-domain ID must fail closed")

    stale = PreparedRoutingIdentity(101, 102, 103, 104, 106, 106)
    try:
        router.query(fixed, identity=stale)
    except ValueError:
        pass
    else:
        raise AssertionError("stale route/value generation must fail closed")
    torch.cuda.synchronize(device)
    print("prepared entity-to-mechanism CUDA correctness passed")


if __name__ == "__main__":
    main()
