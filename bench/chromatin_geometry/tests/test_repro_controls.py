"""Isolated CPU-only checks for immutable capture and GPU budget controls.

All writes use TemporaryDirectory. No live Autobench enrollment, shared budget
ledger, scheduler endpoint, or GPU is touched.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


campaign = import_file("cgs_campaign_tested", REPO / "bench/chromatin_geometry/scripts/campaign.py")
packet = import_file("cgs_packet_tested", REPO / "bench/chromatin_geometry/scripts/run_packet.py")
freezer = import_file("cgs_freezer_tested", REPO / "bench/chromatin_geometry/scripts/freeze_source.py")


def archive_with(entries: list[tuple[str, bytes]]) -> tuple[bytes, str]:
    files = {name: data for name, data in entries}
    manifest = {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in files.items()}
    raw_manifest = json.dumps({"files": manifest}, sort_keys=True).encode() + b"\n"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in [*entries, ("SOURCE-MANIFEST.json", raw_manifest)]:
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    blob = stream.getvalue()
    return blob, hashlib.sha256(blob).hexdigest()


class CaptureTests(unittest.TestCase):
    def test_safe_capture_rejects_symlink_and_omits_runtime_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "include").mkdir()
            (root / "include/api.hh").write_text("immutable")
            (root / "include/.git").mkdir()
            (root / "include/.git/index").write_text("ignored")
            files = {}
            freezer.collect(root, "include", "Source", files)
            self.assertEqual(set(files), {"Source/include/api.hh"})
            (root / "include/link.hh").symlink_to(root / "include/api.hh")
            with self.assertRaisesRegex(RuntimeError, "symlink forbidden"):
                freezer.collect(root, "include", "Source", {})

    def test_extract_binds_archive_manifest_and_rejects_stale_target(self):
        blob, digest = archive_with([("src/a.cc", b"source")])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "capture.tar.gz"
            archive.write_bytes(blob)
            target = root / "fresh"
            packet.extract(archive, target, digest)
            self.assertEqual((target / "src/a.cc").read_bytes(), b"source")
            with self.assertRaises(FileExistsError):
                packet.extract(archive, target, digest)
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                packet.extract(archive, root / "wrong-hash", "0" * 64)

    def test_extract_rejects_path_escape_before_writing_outside_root(self):
        blob, digest = archive_with([("../escape", b"bad")])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "malicious.tar.gz"
            archive.write_bytes(blob)
            target = root / "extract"
            with self.assertRaisesRegex(RuntimeError, "unsafe source archive member"):
                packet.extract(archive, target, digest)
            self.assertFalse((root / "escape").exists())


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        campaign.STUDY = self.root
        campaign.LANES = self.root / "lanes"
        campaign.LEDGER = self.root / "runs/campaign-budget.json"
        lane = campaign.LANES / "0"
        (lane / ".autobench/registry").mkdir(parents=True)
        (lane / ".autobench/config.json").write_text(json.dumps({"active": False}))
        definition = """[benchmark]\nname='fake'\nrevision=1\n[build]\nresources={cpu_cores=2}\n[[packet]]\nname='test'\nkind='correctness'\nkill_seconds=300\nresources={scheduler=[{kind='gpu',mode='exclusive',selectors={resource_id='gpu:v100:GPU-test'}}]}\n"""
        (lane / ".autobench/registry/fake.toml").write_text(definition)
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, lane=0):
        return Namespace(lane=lane)

    def command_for(self, reconcile):
        calls = self.calls
        reconciled = [False]

        def command(argv):
            calls.append(tuple(argv))
            if argv[1] == "status":
                if not reconciled[0]:
                    return {"paused": False, "attempts": []}
                return {"paused": False, "attempts": [{"id": "attempt-1", "job_id": "job-1"}]}
            if argv[1] == "reconcile":
                reconcile(calls)
                reconciled[0] = True
                return {"submitted": 1, "kind": "gpu"}
            raise AssertionError(argv)

        return command

    def test_reserves_before_submission_and_shares_budget_across_lanes(self):
        observed = []

        def reconcile(calls):
            ledger = campaign.load()
            observed.append(ledger["reservations"][-1]["state"])

        campaign.command = self.command_for(reconcile)
        campaign.step(self.args())
        row = campaign.load()["reservations"][-1]
        self.assertEqual(observed, ["reserving"])
        self.assertEqual(row["state"], "submitted")
        self.assertEqual(campaign.charged(campaign.load()), 360)

        lane1 = campaign.LANES / "1"
        (lane1 / ".autobench/registry").mkdir(parents=True)
        (lane1 / ".autobench/config.json").write_text(json.dumps({"active": False}))
        (lane1 / ".autobench/registry/fake.toml").write_text(
            (campaign.LANES / "0/.autobench/registry/fake.toml").read_text())
        campaign.command = self.command_for(lambda calls: None)
        campaign.step(self.args(lane=1))
        row2 = campaign.load()["reservations"][-1]
        self.assertEqual(row2["state"], "submitted")
        self.assertEqual(campaign.charged(campaign.load()), 720)

    def test_unresolved_attempt_blocks_lane_without_reconcile(self):
        campaign.save({"schema": "cgs.budget.v1", "budget_seconds": 14400,
                       "reservations": [{"lane": 0, "state": "unknown", "reserved_seconds": 360}]})
        campaign.command = lambda argv: self.fail("command must not run for unresolved attempt")
        with self.assertRaisesRegex(RuntimeError, "unresolved reservation"):
            campaign.step(self.args())

    def test_aggregate_limit_rejects_before_any_submission(self):
        campaign.save({"schema": "cgs.budget.v1", "budget_seconds": 14400,
                       "reservations": [{"lane": 1, "state": "settled", "charged_seconds": 14100}]})
        def command(argv):
            if argv[1] == "status":
                return {"paused": False, "attempts": []}
            self.fail("GPU submission must not run past aggregate budget")

        campaign.command = command
        with self.assertRaisesRegex(RuntimeError, "GPU budget"):
            campaign.step(self.args())
        self.assertEqual(campaign.charged(campaign.load()), 14100)

    def test_unknown_submission_keeps_full_reservation(self):
        calls = []

        def command(argv):
            calls.append(argv)
            if argv[1] == "status":
                return {"paused": False, "attempts": []}
            raise RuntimeError("injected timeout after submission-capable reconcile")

        campaign.command = command
        with self.assertRaisesRegex(RuntimeError, "injected timeout"):
            campaign.step(self.args())
        ledger = campaign.load()
        self.assertEqual(ledger["reservations"][0]["state"], "unknown")
        self.assertEqual(campaign.charged(ledger), 360)

    def test_unverified_cleanup_and_wrong_uuid_do_not_release_reservation(self):
        row = {"lane": 0, "state": "submitted", "reserved_seconds": 360,
               "job_id": "job-1", "expected_gpu_resource_id": "gpu:v100:GPU-test"}
        campaign.save({"schema": "cgs.budget.v1", "budget_seconds": 14400,
                       "reservations": [row]})

        async def dirty_job(config, job_id):
            return {"job": {"state": "failed", "cleanup_verified": False}}

        campaign.get_job = dirty_job
        campaign.settle(self.args())
        self.assertEqual(campaign.load()["reservations"][0]["state"], "submitted")
        self.assertEqual(campaign.charged(campaign.load()), 360)

        async def wrong_uuid_job(config, job_id):
            return {"job": {"state": "failed", "cleanup_verified": True,
                            "attempt": {"created_at": "2026-11-02T00:00:00Z",
                                        "finished_at": "2026-11-02T00:00:10Z",
                                        "execution_context": {"allocations": [
                                            {"resource_id": "gpu:v100:GPU-other"}]}}}}

        campaign.get_job = wrong_uuid_job
        with self.assertRaisesRegex(RuntimeError, "cannot prove exactly one allocated GPU"):
            campaign.settle(self.args())
        self.assertEqual(campaign.load()["reservations"][0]["state"], "submitted")
        self.assertEqual(campaign.charged(campaign.load()), 360)


if __name__ == "__main__":
    unittest.main()
