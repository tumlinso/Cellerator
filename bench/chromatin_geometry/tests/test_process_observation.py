"""CPU-only checks for serialized NVML PID identity observation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_packet.py'
spec = importlib.util.spec_from_file_location('cgs_process_observation_test', SCRIPT)
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)

DRIVER_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/kda_driver.py'
driver_spec = importlib.util.spec_from_file_location('cgs_process_observation_driver_test', DRIVER_SCRIPT)
driver = importlib.util.module_from_spec(driver_spec)
driver_spec.loader.exec_module(driver)


def wait_until(predicate, description, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f'timed out waiting for {description}')


def proc_state(pid):
    stat = Path(f'/proc/{pid}/stat').read_text()
    return stat[stat.rindex(')') + 2:].split()[0]


def proc_ticks(pid):
    stat = Path(f'/proc/{pid}/stat').read_text()
    return int(stat[stat.rindex(')') + 2:].split()[19])


class ProcessObservationTests(unittest.TestCase):
    def test_short_exit_waits_for_process_query_before_reap(self):
        """A PID reported by NVML stays allocated until its locked /proc read."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_path = root / 'process.lock'
            lock_path.touch()
            events_path = root / 'lifetimes.jsonl'
            marker = root / 'exit-now'
            output = root / 'native-output.json'
            command = [sys.executable, '-c',
                       'import pathlib,sys,time; '
                       'marker=pathlib.Path(sys.argv[1]); '
                       'output=pathlib.Path(sys.argv[2]); '
                       'while_not_ready=True; '
                       'exec("while not marker.exists(): time.sleep(0.002)"); '
                       'output.write_text("{}")', str(marker), str(output)]
            errors = []

            def run_child():
                try:
                    result = driver.run_tracked(command, events_path, lock_path,
                                                {'device_uuid': 'GPU-a', 'binary_sha256': 'a' * 64,
                                                 'backend': 'direct'}, output)
                    if result.returncode != 0:
                        errors.append(f'native child returned {result.returncode}')
                except BaseException as exc:  # propagate worker-thread failures to the test
                    errors.append(repr(exc))

            child_thread = threading.Thread(target=run_child, daemon=True)
            child_thread.start()
            query_entered = threading.Event()
            release_process_query = threading.Event()
            sample_result = []
            sample_errors = []

            def fake_nvidia_run(argv, **kwargs):
                if '--query-gpu=' in ' '.join(argv):
                    return subprocess.CompletedProcess(argv, 0, 'GPU-a, 17\n', '')
                if '--query-compute-apps=' in ' '.join(argv):
                    query_entered.set()
                    if not release_process_query.wait(5):
                        raise AssertionError('test did not release the process query')
                    lines = events_path.read_text().splitlines()
                    start = next(json.loads(line) for line in lines
                                 if json.loads(line).get('event') == 'start')
                    return subprocess.CompletedProcess(
                        argv, 0, f"GPU-a, {start['pid']}, cgs_study\n", '')
                raise AssertionError(f'unexpected nvidia-smi query: {argv}')

            def sample():
                try:
                    sample_result.append(packet.serial_sample(
                        'GPU-a', os.getpid(), lock_path, events_path))
                except BaseException as exc:
                    sample_errors.append(repr(exc))

            try:
                wait_until(lambda: events_path.exists() and events_path.stat().st_size > 0,
                           'native start record')
                start = json.loads(events_path.read_text().splitlines()[0])
                pid = start['pid']
                with patch.object(packet.subprocess, 'run', side_effect=fake_nvidia_run):
                    sampler_thread = threading.Thread(target=sample, daemon=True)
                    sampler_thread.start()
                    self.assertTrue(query_entered.wait(5), 'process query did not start')
                    # The process query now holds the shared flock. Let the short-lived
                    # child exit and wait until it is a zombie held by waitid(WNOWAIT).
                    marker.touch()
                    wait_until(lambda: proc_state(pid) == 'Z', 'native child to exit without reap')
                    observed_ticks = proc_ticks(pid)
                    release_process_query.set()
                    sampler_thread.join(5)
                    self.assertFalse(sampler_thread.is_alive(), 'process query did not finish')
                child_thread.join(5)
                self.assertFalse(child_thread.is_alive(), 'native launcher did not reap the child')
                self.assertEqual(sample_errors, [])
                self.assertEqual(errors, [])
                sample = sample_result[0]
                self.assertEqual(sample['issues'], [])
                self.assertLessEqual(sample['process_query_start_ns'], sample['process_query_end_ns'])
                process = sample['processes'][0]
                self.assertTrue(process['owned'])
                self.assertEqual(process['pid'], pid)
                self.assertEqual(process['proc_start_ticks'], observed_ticks)
                self.assertEqual(process['proc_start_ticks'], start['proc_start_ticks'])
                self.assertEqual(process['invocation_id'], start['invocation_id'])

                end = json.loads(events_path.read_text().splitlines()[1])
                self.assertLessEqual(start['spawn_upper_ns'], sample['process_query_start_ns'])
                self.assertLessEqual(sample['process_query_end_ns'], end['reap_lower_ns'])
                self.assertLessEqual(end['reap_lower_ns'], end['reap_upper_ns'])
                self.assertEqual(end['proc_start_ticks'], process['proc_start_ticks'])
            finally:
                marker.touch(exist_ok=True)
                release_process_query.set()
                child_thread.join(5)

    def test_missing_native_metadata_fails_closed_for_unrelated_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_path = root / 'process.lock'
            lock_path.touch()
            events_path = root / 'lifetimes.jsonl'
            events_path.touch()

            def fake_nvidia_run(argv, **kwargs):
                if '--query-gpu=' in ' '.join(argv):
                    return subprocess.CompletedProcess(argv, 0, 'GPU-a, 0\n', '')
                return subprocess.CompletedProcess(argv, 0,
                                                   f'GPU-a, {os.getppid()}, unrelated\n', '')

            with patch.object(packet.subprocess, 'run', side_effect=fake_nvidia_run):
                sample = packet.serial_sample('GPU-a', os.getpid(), lock_path, events_path)
            self.assertFalse(sample['processes'][0]['owned'])
            self.assertIsNone(sample['processes'][0]['invocation_id'])
            self.assertIn('foreign or unverified GPU compute process observed', sample['issues'])

    def test_pid_with_mismatched_start_ticks_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_path = root / 'process.lock'
            lock_path.touch()
            events_path = root / 'lifetimes.jsonl'
            pid = os.getpid()
            events_path.write_text(json.dumps({
                'event': 'start', 'invocation_id': 'wrong-instance', 'pid': pid,
                'proc_start_ticks': proc_ticks(pid) + 1, 'device_uuid': 'GPU-a'}) + '\n')

            def fake_nvidia_run(argv, **kwargs):
                if '--query-gpu=' in ' '.join(argv):
                    return subprocess.CompletedProcess(argv, 0, 'GPU-a, 0\n', '')
                return subprocess.CompletedProcess(argv, 0, f'GPU-a, {pid}, cgs_study\n', '')

            with patch.object(packet.subprocess, 'run', side_effect=fake_nvidia_run):
                sample = packet.serial_sample('GPU-a', pid, lock_path, events_path)
            process = sample['processes'][0]
            self.assertFalse(process['owned'])
            self.assertEqual(process['invocation_id'], 'wrong-instance')
            self.assertNotEqual(process['proc_start_ticks'],
                                json.loads(events_path.read_text())['proc_start_ticks'])
            self.assertIn('foreign or unverified GPU compute process observed', sample['issues'])


if __name__ == '__main__':
    unittest.main()
