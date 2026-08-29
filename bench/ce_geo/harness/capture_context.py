#!/usr/bin/env python3
"""Capture CE-GEO provenance from files supplied by a leased controller.

This utility performs no device query and runs no benchmark. Device and
topology facts must already have been captured by the external controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def git(repo: Path, *argv: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *argv],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def validate_device(value: dict[str, Any]) -> dict[str, Any]:
    required = ("uuid", "name", "pci_bus_id", "performance_class",
                "driver_version")
    for field in required:
        require(isinstance(value.get(field), str) and bool(value[field]),
                f"device.{field} is missing")
    forbidden = {"environment", "processes", "credentials", "tokens"}
    require(not (forbidden & set(value)), "device capture contains forbidden data")
    return value


def validate_toolchain(value: dict[str, Any]) -> dict[str, Any]:
    for field in ("cxx", "cuda_toolkit", "nvcc", "cmake"):
        require(isinstance(value.get(field), str) and bool(value[field]),
                f"toolchain.{field} is missing")
    return value


def capture(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    require((repo / ".git").exists(), "repo_root is not a Git checkout")
    for path in (args.device_json, args.topology_capture, args.toolchain_json,
                 args.binary, args.cmake_cache):
        require(path.is_file(), f"capture input missing: {path}")

    status = git(repo, "status", "--porcelain=v1", "-z")
    revision = git(repo, "rev-parse", "HEAD").decode().strip()
    submodules = git(repo, "submodule", "status", "--recursive").decode().splitlines()
    device = validate_device(load_object(args.device_json, "device_json"))
    toolchain = validate_toolchain(load_object(args.toolchain_json, "toolchain_json"))
    topology_bytes = args.topology_capture.read_bytes()

    return {
        "schema": "CELLERATOR-CE-GEO-CONTEXT/1",
        "performance_run": False,
        "hardware_query_performed": False,
        "source": {
            "revision": revision,
            "clean": not status,
            "status_digest": sha256_bytes(status),
            "todo_revision": args.todo_revision,
            "submodule_revisions": {
                str(index): line for index, line in enumerate(submodules)
            },
        },
        "device": device,
        "topology": {
            "capture_command": args.topology_command,
            "capture_sha256": sha256_bytes(topology_bytes),
        },
        "toolchain": toolchain,
        "build": {
            "mode": args.build_mode,
            "architecture": args.architecture,
            "binary_sha256": sha256_file(args.binary),
            "cmake_cache_sha256": sha256_file(args.cmake_cache),
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", required=True, type=Path)
    result.add_argument("--todo-revision", required=True)
    result.add_argument("--device-json", required=True, type=Path,
                        help="bounded device JSON captured by the controller")
    result.add_argument("--topology-capture", required=True, type=Path,
                        help="raw topology output captured by the controller")
    result.add_argument("--topology-command", required=True,
                        help="exact argv rendered for provenance only")
    result.add_argument("--toolchain-json", required=True, type=Path)
    result.add_argument("--binary", required=True, type=Path)
    result.add_argument("--cmake-cache", required=True, type=Path)
    result.add_argument("--build-mode", required=True)
    result.add_argument("--architecture", required=True)
    result.add_argument("--output", required=True, type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    value = capture(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("CE-GEO context captured; hardware_query_performed=false performance_run=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
