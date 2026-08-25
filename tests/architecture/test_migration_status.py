#!/usr/bin/env python3
"""Keep the migration status explicit and resistant to completion drift."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROADMAP = (ROOT / "docs" / "migration_roadmap.qmd").read_text()
CURRENT = (ROOT / "docs" / "current_implementation.qmd").read_text()
FOLLOWUPS = (ROOT / "ARCHITECTURE_FOLLOWUPS.md").read_text()


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing migration status contract: {needle}")


def main() -> None:
    require(ROADMAP, "## Live migration status")
    require(ROADMAP, "CE-ARCH-40 through CE-ARCH-79")
    require(ROADMAP, "| 4: Execution Image v2 | partial |")
    require(ROADMAP, "| 5: Instrumentation and corpus | partial |")
    require(ROADMAP, "| 6: Projection plurality | partial |")
    require(ROADMAP, "| 7: Planner and autotuner | partial |")
    require(ROADMAP, "| 8: Objective V2 | partial |")
    require(ROADMAP, "| 9: Native training path | missing |")
    require(ROADMAP, "| 10: Baseplane integration | partial |")
    require(ROADMAP, "| 11: Hierarchy and scale-out | missing |")
    require(ROADMAP, "| CellShard completion boundary | externally blocked |")
    require(ROADMAP, "CE-ARCH-92")

    require(CURRENT, "### CE-ARCH continuation status")
    for task in range(81, 93):
        require(CURRENT, f"CE-ARCH-{task}")
    require(CURRENT, "did not complete the architectural migration")
    require(CURRENT, "CPK1 tightly reflects the v1 operation and projection")
    require(CURRENT, "CPE2 now provides the")

    require(FOLLOWUPS, "CE-ARCH-90")
    require(FOLLOWUPS, "CE-ARCH-91")


if __name__ == "__main__":
    main()
