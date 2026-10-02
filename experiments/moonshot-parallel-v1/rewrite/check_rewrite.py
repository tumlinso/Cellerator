"""Run real CPU rewrite checks and verify source-bound capability evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SUITES = ["test_snapshot", "test_regrowth", "test_checkpoint"]


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.iterdir()) if p.suffix == ".py" or p.name == "README.md"}


def capability():
    return {"route": "experimental CPU Torch controller",
            "supplied_invertible_linear_transform": True,
            "supplied_invariant_quotient": True,
            "off_manifold_state": "rejected",
            "saved_tapes": "explicit acquisition and release; drain before publication or restore",
            "physical_replicas": "coherent gathers; summed adjoints",
            "publication": "validated staged snapshot replacement; external synchronization",
            "regrowth": "seeded nonzero U, initially zero V; externally optimized",
            "optimizer": "external Adam; affected moments reset, scalar steps retained",
            "checkpoint": "fresh controller, residual parameters and external Adam state",
            "native_concurrency": False, "automatic_mechanism_discovery": False,
            "biological_equivalence": False, "cuda_qualification": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    sources = hashes()
    receipt_path = ROOT / "capability-receipt.json"
    if not args.record:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("source_sha256") != sources or receipt.get("capability") != capability():
            raise SystemExit("rewrite capability evidence does not match current source")
        if receipt.get("qualification") != "passed" or receipt.get("suites") != SUITES:
            raise SystemExit("rewrite capability evidence lacks aggregate qualification")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    for suite in SUITES:
        subprocess.run([sys.executable, "-B", "-m", "unittest", "-v", suite],
                       cwd=ROOT, env=env, check=True)
    if hashes() != sources:
        raise SystemExit("rewrite source changed during qualification")
    if args.record:
        import torch
        receipt_path.write_text(json.dumps({"schema_version": 1, "qualification": "passed",
            "source_sha256": sources, "capability": capability(), "suites": SUITES,
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "device": "CPU"}}, indent=2, sort_keys=True) + "\n")
    print("PASS CE-MOON-REWRITE CPU supplied refactoring, live regrowth and checkpoint")


if __name__ == "__main__":
    main()
