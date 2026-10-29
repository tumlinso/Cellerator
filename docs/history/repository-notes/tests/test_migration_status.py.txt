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
    require(ROADMAP, "CE-ARCH-40 through CE-ARCH-92")
    for phase in (
        "4: Execution Image v2", "5: Instrumentation and corpus",
        "6: Projection plurality", "7: Planner and autotuner",
        "8: Objective V2", "9: Native training path",
        "10: Baseplane integration", "11: Hierarchy and scale-out",
    ):
        require(ROADMAP, f"| {phase} | complete |")
    require(ROADMAP, "| CellShard completion boundary | complete for CE-ARCH |")
    require(ROADMAP, "| Migration completion audit | complete |")
    require(ROADMAP, "ce_arch_92_v100_summary.json")
    require(ROADMAP, "490d2ba1-99ce-4d3e-ba1c-65db915a42d1")

    require(CURRENT, "### Closed continuation")
    for task in range(81, 93):
        require(CURRENT, f"CE-ARCH-{task}")
    require(CURRENT, "36 correct candidate results")
    require(CURRENT, "Empirical measurement remains authoritative")
    require(CURRENT, "CPK1 tightly reflects the v1 operation and projection")
    require(CURRENT, "CPE2 provides the")

    require(FOLLOWUPS, "CE-ARCH-90")
    require(FOLLOWUPS, "CE-ARCH-91")


if __name__ == "__main__":
    main()
