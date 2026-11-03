"""Cross-language checks for the unchanged, bounded CFINT001 transport."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from case_io import (HEADER, MAGIC_INPUT, MAX_CASE_SHARD_BYTES, MECHANISM,
                     TYPE_CODE, read_case, verify_case_shards, write_case,
                     write_sharded_case)
from workloads import make_workload


ROOT = Path(__file__).resolve().parents[2]
FILE_LIMIT = 64 * 1024 * 1024


def _probe_command(probe: Path, parts: list[str]) -> list[str]:
    command = [str(probe), "--input", parts[0]]
    for part in parts[1:]:
        command.extend(("--input-shard", part))
    return command


def _million_case_check(directory: Path, probe: Path) -> None:
    # This child is subject to the same inherited per-file cap as an Autobench
    # worker. Its C++ reader emits reconstructed bytes to a pipe, not a file.
    problem, dy = make_workload("correctness_packet_high_fanin_1m",
                                np.float16, np.float16)
    identity = write_sharded_case(directory / "million.case", problem, dy,
                                  np.float32)
    assert identity["bytes"] == 72_000_112
    assert [Path(p).stat().st_size for p in identity["parts"]] == [
        MAX_CASE_SHARD_BYTES, MAX_CASE_SHARD_BYTES, 4_891_248]

    # Independently construct the known fixture's canonical wire hash. This is
    # deliberately separate from the writer and uses no specialized oracle.
    expected = hashlib.sha256()
    expected.update(HEADER.pack(MAGIC_INPUT, 1, 1, 2, 0, 1, 1, 1, 0,
                                1_000_000, 0, TYPE_CODE[np.dtype(np.float16)],
                                TYPE_CODE[np.dtype(np.float16)],
                                TYPE_CODE[np.dtype(np.float32)]))
    for ordinal in range(1_000_000):
        expected.update(MECHANISM.pack(ordinal, 0, 0, 0, 2, 0, 0, 1, 0, 0))
    for values, dtype in ((problem.x, "<f2"), (problem.z, "<f2"),
                          (problem.alpha, "<f2"), (problem.bias, "<f2"),
                          (problem.gamma, "<f2"), (dy, "<f4")):
        expected.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
    assert identity["sha256"] == expected.hexdigest()
    native = subprocess.run(_probe_command(probe, identity["parts"]),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=directory, check=True, timeout=60)
    assert len(native.stdout) == identity["bytes"]
    assert hashlib.sha256(native.stdout).hexdigest() == expected.hexdigest()
    verify_case_shards(identity)


class ShardedTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._build_dir = tempfile.TemporaryDirectory(prefix="factorized-wire-probe-")
        cls.probe = Path(cls._build_dir.name) / "wire_input_probe"
        subprocess.run([
            "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-I", str(ROOT / "bench" / "factorized_interactions"),
            str(ROOT / "tests" / "factorized_interactions" / "wire_input_probe.cc"),
            "-o", str(cls.probe),
        ], cwd=cls._build_dir.name, check=True, capture_output=True, text=True,
            timeout=60)

    @classmethod
    def tearDownClass(cls):
        cls._build_dir.cleanup()

    def test_reconstructs_identical_wire_across_headers_roles_and_terms(self):
        problem, dy = make_workload("correctness_ordered_role_context_variants",
                                    np.float16, np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.case"
            original_identity = write_case(original, problem, dy, np.float32)
            identity = write_sharded_case(root / "split.case", problem, dy,
                                          np.float32, shard_bytes=73)
            self.assertGreater(len(identity["parts"]), 2)
            self.assertEqual(identity["sha256"], original_identity["sha256"])
            actual = subprocess.run(_probe_command(self.probe, identity["parts"]),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    cwd=root, check=True, timeout=10).stdout
            self.assertEqual(actual, original.read_bytes())
            restored, restored_dy, dtype = read_case(original)
            self.assertEqual(restored.mechanisms, problem.mechanisms)
            np.testing.assert_array_equal(restored_dy, dy)
            self.assertEqual(dtype, np.dtype(np.float32))

    def test_rejects_corrupted_reordered_and_missing_shards(self):
        problem, dy = make_workload("correctness_ordered_role_context_variants")
        with tempfile.TemporaryDirectory() as directory:
            identity = write_sharded_case(Path(directory) / "split.case", problem,
                                          dy, shard_bytes=73)
            swapped = dict(identity, parts=list(identity["parts"]))
            swapped["parts"][0], swapped["parts"][1] = (
                swapped["parts"][1], swapped["parts"][0])
            with self.assertRaisesRegex(ValueError, "frozen identity"):
                verify_case_shards(swapped)
            part = Path(identity["parts"][0])
            data = bytearray(part.read_bytes())
            data[0] ^= 1
            part.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "frozen identity"):
                verify_case_shards(identity)
            missing = Path(identity["parts"][-1])
            missing.unlink()
            result = subprocess.run(_probe_command(self.probe, identity["parts"]),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    cwd=directory, check=False, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"cannot open case input", result.stderr)

    @unittest.skipUnless(os.name == "posix", "requires POSIX file-size limit")
    def test_full_million_case_under_worker_file_limit(self):
        with tempfile.TemporaryDirectory(prefix="factorized-million-case-") as directory:
            def limit_file_size():
                resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_LIMIT, FILE_LIMIT))

            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--million-case",
                 directory, str(self.probe)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=directory, preexec_fn=limit_file_size, check=False, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--million-case":
        _million_case_check(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        unittest.main()
