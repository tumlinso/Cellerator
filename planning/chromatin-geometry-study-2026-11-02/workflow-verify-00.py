#!/usr/bin/env python3
"""Validate the captured CGS-00 provisioning evidence in this source tree.

The status snapshot describes the instant of capture; it is not a claim that
Scheduler remains available later. The authoritative archive and baseline
remain in the study workspace; this bundle captures their exact bytes for the
source-bound verification provider.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "workflow-evidence"
EXPECTED_ARCHIVE = "f2d018e85046e41d1dc8b5f0895ea52f3465007addad1fea600e85aa44080bec"
EXPECTED_PLAN = "d42e1195be751597010af0bb8ca1ba3b818ffe87aca848e4e15a60111f68749a"
EXPECTED_FILES = {
    "baseline-source.tar.gz", "source-baseline.json", "baseline-archive.json",
    "kda-draft.md", "resource-status.json", "control-summary.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(name):
    return json.loads((BUNDLE / name).read_text(encoding="utf-8"))


def main():
    manifest = load("manifest.json")
    require(set(manifest["files"]) == EXPECTED_FILES, "evidence inventory differs")
    for name, expected in manifest["files"].items():
        require(digest(BUNDLE / name) == expected, f"captured bytes differ: {name}")

    baseline = load("source-baseline.json")
    closure = load("baseline-archive.json")
    require(closure["source_head"] == baseline["head"], "source heads differ")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    require(head == baseline["head"], "verification workspace has the wrong source base")
    require(len(baseline["dirty"]) == 37, "dirty-file baseline count differs")
    require(closure["archive_sha256"] == EXPECTED_ARCHIVE, "archive manifest differs")
    require(digest(BUNDLE / "baseline-source.tar.gz") == EXPECTED_ARCHIVE, "archive hash differs")
    with tarfile.open(BUNDLE / "baseline-source.tar.gz", "r:gz") as tar:
        members = tar.getmembers()
        require(len(members) == closure["file_count"] == 1084, "archive member count differs")
        geometry = tar.extractfile("cellular_geometry.md")
        require(geometry is not None, "protected geometry missing from closure")
        require(hashlib.sha256(geometry.read()).hexdigest() == baseline["protected_geometry_sha256"],
                "protected geometry differs from baseline")

    draft = (BUNDLE / "kda-draft.md").read_text(encoding="utf-8")
    require(len(draft) > 1000 and "chromatin" in draft.lower(), "KDA draft missing")
    control = load("control-summary.json")
    require(control["plan_change_digest"] == EXPECTED_PLAN, "provisioning plan receipt differs")
    require(control["plan_revision"] == "accepted-" + EXPECTED_PLAN, "plan revision differs")
    require(control["run_id"] == "cgs-chromatin-study-20261102-v1", "run identity differs")

    resource = load("resource-status.json")
    doctor = resource["study_doctor"]
    require(bool(resource["captured_at"]), "status capture time missing")
    require(doctor["initialized"] and doctor["scheduler_configured"]
            and doctor["scheduler_connected"] and doctor["scheduler_access"]["ready"],
            "study Scheduler was not ready at capture")
    require(resource["study_status"]["registrations"] == [], "study enrollment not empty at capture")
    require(resource["parent_status"]["paused"] is True, "parent pause not preserved")
    require(resource["aggregate_gpu_seconds"] == 14400 and resource["max_job_seconds"] == 900,
            "resource budget differs")
    print("CGS-00 captured source and provisioning evidence verified")


if __name__ == "__main__":
    main()
