#!/usr/bin/env python3
"""Export one already-imported Autobench packet receipt and report identity."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--report-relative", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rel = PurePosixPath(args.report_relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts or rel.as_posix() != args.report_relative:
        raise ValueError("report path must be relative to the imported artifact")
    result = subprocess.run(["autobench", "results", "--project", str(args.project),
                             args.benchmark, "--details", "--limit", "100"],
                            check=True, capture_output=True, text=True)
    rows = json.loads(result.stdout)[args.benchmark]
    matched = [row for row in rows if row.get("id") == args.attempt]
    if len(matched) != 1:
        raise ValueError("attempt is absent or ambiguous in official Autobench results")
    row = matched[0]
    payload = json.loads(row["result_json"])
    job, worker = payload["job"], payload["worker"]
    if row["state"] != "succeeded" or job.get("state") not in ("succeeded", "completed") or \
            job.get("cleanup_verified") is not True or worker.get("status") != "succeeded" or \
            worker.get("returncode") != 0 or worker.get("scheduler_job_id") != row["job_id"]:
        raise ValueError("attempt lacks a successful worker and verified Scheduler cleanup")
    root = args.project.resolve() / ".autobench" / "evidence" / args.benchmark / str(row["packet_index"])
    pointer = json.loads((root / "current.json").read_text())
    if pointer.get("attempt") != args.attempt or pointer.get("cleanup_verified") is not True or \
            pointer.get("scheduler_job_id") != row["job_id"] or pointer.get("fingerprint") != row["fingerprint"]:
        raise ValueError("current imported evidence does not match the requested attempt")
    report = root / row["fingerprint"] / rel
    if report.is_symlink() or not report.is_file() or not report.resolve().is_relative_to(root.resolve()):
        raise ValueError("report is missing or escaped imported evidence")
    wrapper = {"schema": "cellerator.factorized_interactions.finished_receipt.v1",
               "autobench_results_row": row, "current_pointer": pointer,
               "report_path": str(report), "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(wrapper, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"attempt": args.attempt, "report_sha256": wrapper["report_sha256"],
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
