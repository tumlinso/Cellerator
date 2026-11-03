#!/usr/bin/env python3
"""Reproduce snapshot eviction with a tiny private in-memory authority stub.

Run with Autobench's installed Python. All files used here belong to a fresh
temporary directory; no Scheduler, real Autobench database or workspace is used.
Use --patched only against a reviewed patched copy of autobench on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import autobench.engine as engine_module


class EmptyDb:
    def execute(self, _query):
        return []


class StoreStub:
    db = EmptyDb()

    def get(self, key, default=None):
        return 1024 if key == "retained_cap_bytes" else default

    def unresolved(self):
        return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patched", action="store_true")
    parser.add_argument("--verify-patch", action="store_true",
                        help="also apply the proposed patch to a temporary package copy")
    args = parser.parse_args()
    if args.patched and args.verify_patch:
        parser.error("--patched and --verify-patch cannot be combined")
    if args.verify_patch:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        original = subprocess.run([sys.executable, __file__], env=env, text=True,
                                  capture_output=True, check=True, timeout=20)
        with tempfile.TemporaryDirectory(prefix="autobench-pin-patch-test-") as directory:
            root = Path(directory)
            package = Path(inspect.getfile(engine_module)).parent
            shutil.copytree(package, root / "src/autobench",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            patch = Path(__file__).with_name("autobench-snapshot-pin.patch").resolve()
            subprocess.run(["patch", "-p1", "-i", str(patch)], cwd=root,
                           text=True, capture_output=True, check=True, timeout=20)
            env["PYTHONPATH"] = str(root / "src")
            patched = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                      "--patched"], cwd=root, env=env, text=True,
                                     capture_output=True, check=True, timeout=20)
            print(json.dumps({"installed": json.loads(original.stdout),
                              "temporary_patch": json.loads(patched.stdout)},
                             sort_keys=True))
        return 0
    with tempfile.TemporaryDirectory(prefix="autobench-snapshot-eviction-repro-") as directory:
        auto = Path(directory) / ".autobench"
        state = auto / "state"
        snapshot = state / "snapshots" / ("a" * 64)
        snapshot.mkdir(parents=True)
        (snapshot / "captured-source").write_bytes(b"s" * 200)
        data = auto / "data"
        data.mkdir()
        (data / "retained-data").write_bytes(b"d" * 600)

        engine = engine_module.Engine.__new__(engine_module.Engine)
        engine.autobench = auto
        engine.state = state
        engine.store = StoreStub()
        engine.limits = {"retained_cap_bytes": 1024}
        initial = engine_module._disk_usage(auto)
        reservation = 256
        if initial + reservation <= 1024:
            raise AssertionError("test does not reach the reservation-pressure path")
        if args.patched:
            engine._evict_safe(required_free=reservation, protected_paths=(snapshot,))
        else:
            engine._evict_safe(required_free=reservation)
        remaining = engine_module._disk_usage(auto)
        exists = snapshot.is_dir()
        admitted_by_quota = remaining + reservation <= 1024
        print(json.dumps({
            "engine_module": inspect.getfile(engine_module),
            "initial_bytes": initial,
            "remaining_bytes": remaining,
            "reservation_bytes": reservation,
            "snapshot_exists_after_eviction": exists,
            "quota_would_allow_submission": admitted_by_quota,
            "patched": args.patched,
        }, sort_keys=True))
        if args.patched:
            return 0 if exists and not admitted_by_quota else 1
        return 0 if not exists and admitted_by_quota else 1


if __name__ == "__main__":
    raise SystemExit(main())
