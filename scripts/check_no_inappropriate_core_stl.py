#!/usr/bin/env python3
"""Reject new controlled STL ownership in Cellerator production roots."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


FAMILIES = (
    "vector",
    "map",
    "unordered_map",
    "set",
    "unordered_set",
    "priority_queue",
    "shared_ptr",
    "deque",
    "list",
)
SOURCE_SUFFIXES = {".h", ".hh", ".hpp", ".cuh", ".cc", ".cpp", ".cu"}
PRODUCTION_ROOTS = ("include/Cellerator", "src")

# Existing migration debt only. Values are non-increasing ceilings, not quotas.
# Every entry names the semantic migration lane; no entry authorizes a new owner.
ALLOWLIST: dict[str, dict[str, tuple[int, str]]] = {
    "include/Cellerator/compute/operators/graph/forward_candidates.cuh": {"vector": (3, "CE-PTR-10 bounded edge table")},
    "include/Cellerator/compute/operators/graph/forward_prune.cuh": {"vector": (10, "CE-PTR-10 fixed-width graph relation")},
    "include/Cellerator/compute/operators/graph/incremental_insert.cuh": {"vector": (5, "CE-PTR-10 slab assignment relation")},
    "include/Cellerator/compute/operators/graph/record_table.cuh": {"vector": (12, "CE-PTR-10 trajectory record image")},
    "include/Cellerator/compute/operators/graph/slab_index.cuh": {"vector": (7, "CE-PTR-10 embryo span table")},
    "include/Cellerator/compute/operators/graph/supernode_reduce.cuh": {"vector": (27, "CE-PTR-10 topology and member images")},
    "include/Cellerator/compute/sampling.hh": {"vector": (7, "CE-PTR-07 sample image and workspace")},
    "include/Cellerator/examples/trajectory/trajectory_build.cuh": {"vector": (1, "CE-PTR-10 example-facing graph migration")},
    "include/Cellerator/examples/trajectory/trajectory_query.cuh": {"vector": (6, "CE-PTR-10 caller-owned query views")},
    "include/Cellerator/geometry/gating.hh": {"vector": (2, "CE-PTR-04 region selection table")},
    "include/Cellerator/geometry/gating_cuda.cuh": {"vector": (4, "CE-PTR-04 compiled route image")},
    "include/Cellerator/geometry/layout_metrics.hh": {"vector": (1, "CE-PTR-04 exact metrics table")},
    "include/Cellerator/geometry/layout_selector.hh": {"vector": (1, "CE-PTR-04 region selection table")},
    "include/Cellerator/geometry/pack.hh": {"vector": (4, "CE-PTR-04 direct image compilation")},
    "include/Cellerator/geometry/planner.hh": {"vector": (11, "CE-PTR-03 static plan image")},
    "include/Cellerator/runtime/device_buffer.cuh": {"shared_ptr": (2, "CE-PTR-13 session allocation handles")},
    "src/compute/dataset/sampling.cc": {"vector": (18, "CE-PTR-07 sample workspaces"), "priority_queue": (2, "CE-PTR-07 bounded selection")},
    "src/compute/dataset/sampling_materialization.cc": {"vector": (5, "CE-PTR-07 sampled CSR image")},
    "src/compute/neighbors/scoring/cuvs_sharded_knn.cu": {"vector": (1, "CE-PTR-12 move or retire legacy staging")},
    "src/compute/projection/physical_feature_major.cc": {"vector": (1, "CE-PTR-14 projection builder workspace")},
    "src/compute/projection/physical_transpose.cc": {"vector": (2, "CE-PTR-14 projection builder workspace")},
    "src/geometry/candidate_discovery/gene_candidate_discovery.cc": {"vector": (5, "CE-PTR-09 resident candidate pipeline")},
    "src/geometry/candidate_relation.cc": {"vector": (2, "CE-PTR-04 packed candidate relation")},
    "src/geometry/layout_metrics.cc": {"vector": (1, "CE-PTR-04 metrics workspace")},
    "src/geometry/optimizer.cc": {"vector": (2, "CE-PTR-05/06 prepared optimizer tables")},
    "src/geometry/optimizer_state.hh": {"vector": (20, "CE-PTR-05 bounded state representation")},
    "src/geometry/planner.cc": {"vector": (7, "CE-PTR-03 static plan image")},
}


def strip_comments_and_literals(text: str) -> str:
    token = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.DOTALL)
    return token.sub(lambda match: "\n" * match.group(0).count("\n"), text)


def occurrences(path: pathlib.Path) -> dict[str, int]:
    text = strip_comments_and_literals(path.read_text(encoding="utf-8", errors="replace"))
    return {
        family: len(re.findall(rf"\bstd::{family}\s*<", text))
        for family in FAMILIES
    }


def audit(repo_root: pathlib.Path) -> list[str]:
    failures: list[str] = []
    seen: set[str] = set()
    for root_name in PRODUCTION_ROOTS:
        root = repo_root / root_name
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            relative = path.relative_to(repo_root).as_posix()
            counts = occurrences(path)
            allowed = ALLOWLIST.get(relative, {})
            for family, count in counts.items():
                if count == 0:
                    continue
                seen.add(relative)
                if family not in allowed:
                    failures.append(f"{relative}: unallowlisted std::{family} owner ({count})")
                    continue
                ceiling, rationale = allowed[family]
                if count > ceiling:
                    failures.append(
                        f"{relative}: std::{family} count {count} exceeds ceiling {ceiling} ({rationale})"
                    )
            for family in allowed:
                if counts.get(family, 0) == 0:
                    # Debt removal is accepted; stale entries are reported only in strict mode.
                    continue
    return failures


def stale_entries(repo_root: pathlib.Path) -> list[str]:
    stale: list[str] = []
    for relative, families in ALLOWLIST.items():
        path = repo_root / relative
        counts = occurrences(path) if path.is_file() else {}
        for family, (_, rationale) in families.items():
            if counts.get(family, 0) == 0:
                stale.append(f"{relative}: stale std::{family} allowlist entry ({rationale})")
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--strict-stale", action="store_true", help="also reject allowlist entries whose debt has been removed")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    failures = audit(repo_root)
    if args.strict_stale:
        failures.extend(stale_entries(repo_root))
    if failures:
        for failure in failures:
            print(f"CE-PTR STL policy violation: {failure}", file=sys.stderr)
        return 1
    print("CE-PTR STL policy: production ownership is within non-increasing allowlist ceilings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
