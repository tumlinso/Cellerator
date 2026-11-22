#!/usr/bin/env python3
"""Validate the authoritative CE-GEO execution program structure."""

import pathlib
import sys


REQUIRED_REFERENCES = (
    "[AGENTS.md](../AGENTS.md)",
    "[scope.md](../scope.md)",
    "[Architecture](architecture.qmd)",
    "[Current Implementation](current_implementation.qmd)",
    "[Migration Roadmap](migration_roadmap.qmd)",
)

REQUIRED_SECTIONS = (
    "## Authority and purpose",
    "### Settled contracts and empirical questions",
    "## Settled execution architecture",
    "## Exact covers and identity",
    "## Portable CSG1 artifact",
    "## Device-specific CPE2 artifact",
    "## Work window and admissibility",
    "## Hardware, capabilities, and providers",
    "## Candidate catalog and executable program",
    "## Geometry strategy and support evidence",
    "## Volta execution",
    "## Biology-centered relation algebra",
    "## Numerical policy",
    "## Complete-cost evidence",
    "## STL coexistence",
    "## Task graph and integration",
    "## Gates and allowed outcomes",
    "## Completion semantics",
    "## Permission-gated CE-AMP extension",
)

REQUIRED_TERMS = (
    "semantic cover",
    "physical contribution cover",
    "CE-GEO-VOLTA-COMPLETE",
    "CE-GEO-COMPLETE",
    "CE-AMP-PERMISSION",
    "not_granted",
    "explicit human authorization",
    "CE-PTR",
    "cuda-benchmark-mutex",
    "planner is the sole final candidate-selection authority",
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_program.py <program.md>", file=sys.stderr)
        return 2

    try:
        text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    except OSError as error:
        print(f"CE-GEO program validation failed: {error}", file=sys.stderr)
        return 1

    missing = [item for item in REQUIRED_REFERENCES + REQUIRED_SECTIONS + REQUIRED_TERMS if item not in text]
    if missing:
        print("CE-GEO program validation failed; missing:", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        return 1

    if "architecture is undecided" in text.lower():
        print("CE-GEO program validation failed: settled architecture reopened", file=sys.stderr)
        return 1

    print("CE-GEO program validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
