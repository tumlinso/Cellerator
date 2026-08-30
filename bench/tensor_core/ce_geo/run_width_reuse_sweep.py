#!/usr/bin/env python3
"""Build, run, profile, and validate the leased CE-GEO-114 sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "build/ceGeoWidthReuseSweep"
NCU_METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
]


def run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def profiler_record() -> dict[str, object]:
    ncu = shutil.which("ncu")
    if ncu is None:
        return {
            "record_type": "profiler",
            "campaign_id": "width-reuse",
            "profiler_available": False,
            "metrics": {},
            "limitation": "Nsight Compute executable ncu is unavailable; cache and stall counters were not inferred.",
        }
    result = run([
        ncu, "--target-processes", "all", "--csv", "--page", "raw",
        "--metrics", ",".join(NCU_METRICS), str(BINARY), "--profile-only",
    ], check=False)
    values: dict[str, object] = {}
    rows = list(csv.reader(result.stdout.splitlines()))
    header_index = next((index for index, row in enumerate(rows)
                         if "Metric Name" in row and "Metric Value" in row), None)
    if header_index is not None:
        header = rows[header_index]
        name_index = header.index("Metric Name")
        value_index = header.index("Metric Value")
        unit_index = header.index("Metric Unit") if "Metric Unit" in header else None
        for row in rows[header_index + 1:]:
            if len(row) <= max(name_index, value_index):
                continue
            name = row[name_index]
            if name not in NCU_METRICS:
                continue
            raw = row[value_index].replace(",", "")
            try:
                value: object = float(raw)
            except ValueError:
                value = raw
            values[name] = {
                "value": value,
                "unit": row[unit_index] if unit_index is not None and len(row) > unit_index else "",
            }
    record: dict[str, object] = {
        "record_type": "profiler",
        "campaign_id": "width-reuse",
        "profiler_available": True,
        "profiler_returncode": result.returncode,
        "metrics": values,
    }
    missing = [metric for metric in NCU_METRICS if metric not in values]
    if result.returncode != 0 or missing:
        detail = result.stderr.strip().splitlines()
        record["limitation"] = (
            "Nsight Compute did not return every requested counter; no missing "
            "cache or stall value was inferred."
        )
        record["missing_metrics"] = missing
        record["profiler_diagnostic"] = detail[-1][:400] if detail else "no diagnostic"
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    compile_result = run([
        "nvcc", "-std=c++17", "-arch=sm_70", "-O3", "-lineinfo",
        "-Xcompiler=-Wall,-Wextra,-Werror", "-I.", "-Iinclude",
        "bench/tensor_core/ce_geo/width_reuse_sweep.cu",
        "-lcudart", "-o", str(BINARY),
    ])
    if compile_result.stderr:
        print(compile_result.stderr, file=sys.stderr, end="")
    run_result = run([
        str(BINARY), "--output", str(output), "--warmups", "1",
        "--repeats", "3",
    ])
    if run_result.stderr:
        print(run_result.stderr, file=sys.stderr, end="")
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(profiler_record(), sort_keys=True) + "\n")
    validated = run([
        sys.executable, "bench/ce_geo/harness/run.py", "--campaign",
        "width-reuse", "--output", str(output),
    ])
    print(json.dumps(json.loads(validated.stdout), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
