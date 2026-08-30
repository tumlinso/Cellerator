#!/usr/bin/env python3
"""Audit only manifest-declared CE-GEO production files and frozen images."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any


SOURCE_SUFFIXES = {".h", ".hh", ".hpp", ".cuh", ".cc", ".cpp", ".cu"}
ALLOWED_PRODUCTION_ROOTS = (
    "include/Cellerator/compute/architecture/",
    "include/Cellerator/compute/operation/",
    "include/Cellerator/geometry/",
    "src/compute/architecture/",
    "src/compute/projection/",
    "src/geometry/compiler/",
    "src/geometry/persistence/",
)
ALLOWED_EXACT_PATHS = {"include/Cellerator/runtime/device_descriptor.hh"}
PHYSICAL_PROVIDER_ROOT = "src/compute/architecture/providers/"
VALID_LAYERS = {
    "architecture_contract",
    "operation_contract",
    "geometry",
    "csg1",
    "physical_provider",
    "physical_projection",
    "compiler",
    "persistence",
}
STL_OWNERS = (
    "string",
    "basic_string",
    "vector",
    "map",
    "unordered_map",
    "set",
    "unordered_set",
    "priority_queue",
    "shared_ptr",
    "unique_ptr",
    "deque",
    "list",
)

TOKEN = re.compile(
    r'//[^\n]*|/\*.*?\*/|R"(?P<raw_tag>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=raw_tag)"'
    r'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
    re.DOTALL,
)
WMMA = re.compile(r"\b(?:nvcuda\s*::\s*)?wmma\b|\bmma_sync\s*\(")
ARCHITECTURE = re.compile(
    r"\b(?:sm_?\d+|volta|ampere|hopper|blackwell|nvidia|amd|gpu|device|"
    r"architecture|compute_(?:major|minor|capability)|warp(?:_size)?|"
    r"tensor_core|cuda\w*)\b",
    re.IGNORECASE,
)
FAST_MATH_FLAGS = re.compile(r"--use_fast_math|-ffast-math")
FAST_MATH_CODE = re.compile(
    r"\b__FAST_MATH__\b|"
    r"\b__(?:exp|exp2|exp10|log|log2|log10|pow|sinf|cosf|tanf|fdivide)f\s*\("
)
ATOMIC = re.compile(r"\b(?:atomic[A-Z]\w*|std\s*::\s*atomic|cuda\s*::\s*atomic)\b")


class ManifestError(ValueError):
    """The manifest cannot define a closed CE-GEO audit."""


@dataclass(frozen=True)
class ProductionEntry:
    path: str
    layer: str


def strip_comments_and_literals(text: str) -> str:
    """Preserve line count while removing tokens that cannot affect code."""

    return TOKEN.sub(lambda match: "\n" * match.group(0).count("\n"), text)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def clean_relative_path(value: Any, field: str) -> str:
    require(isinstance(value, str) and value != "", f"{field} must be a path")
    path = pathlib.PurePosixPath(value)
    require(not path.is_absolute(), f"{field} must be repository-relative")
    require(".." not in path.parts, f"{field} escapes the repository")
    normalized = path.as_posix()
    require(normalized == value, f"{field} is not normalized: {value}")
    require("CE-PTR" not in value.upper().replace("_", "-"),
            f"{field} names CE-PTR, which this audit must not certify")
    return normalized


def allowed_production_path(path: str) -> bool:
    return path in ALLOWED_EXACT_PATHS or any(
        path.startswith(root) for root in ALLOWED_PRODUCTION_ROOTS
    )


def parse_manifest(document: Any) -> tuple[list[ProductionEntry], list[dict[str, str]]]:
    require(isinstance(document, dict), "manifest root must be an object")
    require(document.get("schema_version") == 1, "unsupported schema_version")
    require(document.get("campaign") == "CE-GEO", "manifest must certify CE-GEO only")
    require(document.get("scope") == "owned-production-only",
            "manifest scope must be owned-production-only")
    require(set(document) == {
        "schema_version", "campaign", "scope", "production", "protected_compatibility"
    }, "manifest contains unknown or missing top-level fields")

    production = document["production"]
    require(isinstance(production, list) and production,
            "production inventory must be a nonempty array")
    entries: list[ProductionEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(production):
        require(isinstance(item, dict) and set(item) == {"path", "layer"},
                f"production[{index}] must contain exactly path and layer")
        path = clean_relative_path(item["path"], f"production[{index}].path")
        layer = item["layer"]
        require(layer in VALID_LAYERS,
                f"production[{index}].layer is invalid: {layer}")
        require(path not in seen, f"duplicate production path: {path}")
        require(allowed_production_path(path),
                f"broad production ownership is outside CE-GEO roots: {path}")
        require(pathlib.PurePosixPath(path).suffix in SOURCE_SUFFIXES
                or pathlib.PurePosixPath(path).name == "CMakeLists.txt",
                f"production path is not C++ or CUDA source: {path}")
        if layer == "physical_provider":
            require(path.startswith(PHYSICAL_PROVIDER_ROOT),
                    f"physical_provider is outside provider ownership: {path}")
        else:
            require(not path.startswith(PHYSICAL_PROVIDER_ROOT),
                    f"provider source must declare physical_provider layer: {path}")
        seen.add(path)
        entries.append(ProductionEntry(path, layer))

    protected = document["protected_compatibility"]
    require(isinstance(protected, list) and protected,
            "protected_compatibility must be a nonempty array")
    protected_seen: set[str] = set()
    contracts: set[str] = set()
    normalized_protected: list[dict[str, str]] = []
    for index, item in enumerate(protected):
        require(isinstance(item, dict)
                and set(item) == {"path", "contract", "sha256"},
                f"protected_compatibility[{index}] has invalid fields")
        path = clean_relative_path(
            item["path"], f"protected_compatibility[{index}].path")
        contract = item["contract"]
        digest = item["sha256"]
        require(contract in {"CPK1", "CPE2"},
                f"unknown compatibility contract: {contract}")
        require(isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"invalid sha256 for protected path: {path}")
        require(path not in protected_seen, f"duplicate protected path: {path}")
        require(allowed_production_path(path),
                f"protected path is outside CE-GEO read roots: {path}")
        protected_seen.add(path)
        contracts.add(contract)
        normalized_protected.append(
            {"path": path, "contract": contract, "sha256": digest}
        )
    require(contracts == {"CPK1", "CPE2"},
            "protected compatibility inventory must pin both CPK1 and CPE2")
    return entries, normalized_protected


def source_violations(entry: ProductionEntry, text: str) -> list[str]:
    code = strip_comments_and_literals(text)
    failures: list[str] = []
    for family in STL_OWNERS:
        count = len(re.findall(rf"\bstd\s*::\s*{family}\s*<", code))
        if count:
            failures.append(f"new STL ownership std::{family} ({count})")
    if re.search(r"\b(?:operator\s+)?new\s*(?:\[|\()", code):
        failures.append("dynamic allocation with new")
    if WMMA.search(code) and entry.layer != "physical_provider":
        failures.append("WMMA leaked outside physical provider implementation")
    if entry.layer == "csg1" and ARCHITECTURE.search(code):
        failures.append("architecture detail leaked into portable CSG1")
    cmake_fast_math = pathlib.PurePosixPath(entry.path).name == "CMakeLists.txt" \
        and FAST_MATH_FLAGS.search(text)
    if cmake_fast_math or FAST_MATH_FLAGS.search(code) or FAST_MATH_CODE.search(code):
        failures.append("global or intrinsic fast-math policy")
    if ATOMIC.search(code):
        failures.append("atomic operation in CE-GEO owned production")
    return failures


def resolve_file(repo_root: pathlib.Path, relative: str) -> pathlib.Path:
    path = repo_root / relative
    require(path.is_file(), f"manifest path does not exist: {relative}")
    require(not path.is_symlink(), f"manifest path may not be a symlink: {relative}")
    require(path.resolve().is_relative_to(repo_root.resolve()),
            f"manifest path escapes repository: {relative}")
    return path


def audit_manifest(manifest_path: pathlib.Path,
                   repo_root: pathlib.Path) -> tuple[list[str], int, int]:
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        production, protected = parse_manifest(document)
        failures: list[str] = []
        for entry in production:
            path = resolve_file(repo_root, entry.path)
            for violation in source_violations(
                    entry, path.read_text(encoding="utf-8", errors="replace")):
                failures.append(f"{entry.path}: {violation}")
        for item in protected:
            path = resolve_file(repo_root, item["path"])
            actual = sha256(path)
            if actual != item["sha256"]:
                failures.append(
                    f"{item['path']}: frozen {item['contract']} sha256 mismatch "
                    f"(expected {item['sha256']}, actual {actual})"
                )
        return failures, len(production), len(protected)
    except (OSError, json.JSONDecodeError, ManifestError) as error:
        return [f"manifest: {error}"], 0, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--repo-root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    failures, production_count, protected_count = audit_manifest(
        args.manifest.resolve(), args.repo_root.resolve()
    )
    if failures:
        for failure in failures:
            print(f"CE-GEO static contract violation: {failure}", file=sys.stderr)
        return 1
    print(
        "CE-GEO static contracts passed: "
        f"production={production_count} protected={protected_count} "
        "campaign=CE-GEO (CE-PTR not certified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
