import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / 'scripts/run_packet.py'
spec = importlib.util.spec_from_file_location('cgs_run_packet_telemetry', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SerialTelemetryTest(unittest.TestCase):
    def sample(self, gpu, apps, owned=True, rc=0):
        outputs = [SimpleNamespace(stdout=gpu, returncode=rc), SimpleNamespace(stdout=apps, returncode=0)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock, events = root / 'lock', root / 'events'
            lock.touch()
            events.write_text(json.dumps({'event': 'start', 'invocation_id': 'one', 'pid': 42,
                                          'device_uuid': 'GPU-a', 'proc_start_ticks': 123}) + '\n')
            with patch.object(module.subprocess, 'run', side_effect=outputs), \
                 patch.object(module, 'is_descendant', return_value=owned), \
                 patch.object(module, 'proc_start_ticks', return_value=123):
                return module.serial_sample('GPU-a', 10, lock, events)

    def test_clean_other_gpu_and_owned_target_process(self):
        x = self.sample('GPU-a, 30\nGPU-b, 0\n', 'GPU-a, 42, cgs\n')
        self.assertEqual(x['issues'], [])

    def test_malformed_lifetime_record_fails_closed(self):
        outputs = [SimpleNamespace(stdout='GPU-a, 0\n', returncode=0),
                   SimpleNamespace(stdout='', returncode=0)]
        with tempfile.TemporaryDirectory() as tmp:
            lock, events = Path(tmp) / 'lock', Path(tmp) / 'events'
            lock.touch()
            events.write_text('[]\n')
            with patch.object(module.subprocess, 'run', side_effect=outputs):
                result = module.serial_sample('GPU-a', 10, lock, events)
            self.assertTrue(result['issues'])

    def test_foreign_or_dirty_or_missing_data_rejected(self):
        for x in (self.sample('GPU-a, 20\nGPU-b, 5\n', ''),
                  self.sample('GPU-a, 20\nGPU-b, 0\n', 'GPU-a, 42, other\n', owned=False),
                  self.sample('GPU-b, 0\n', ''), self.sample('GPU-a, 0\n', '', rc=1)):
            self.assertTrue(x['issues'])

    def test_immediate_exit_has_no_during_sample(self):
        sample = {'devices': [{'uuid': 'GPU-a', 'utilization_percent': 0}], 'processes': [], 'issues': []}
        process = SimpleNamespace(pid=5, returncode=0, poll=lambda: 0, communicate=lambda: ('', ''))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(module.os.environ, {'AUTOBENCH_SCHEDULER_JOB_ID': 'job'}), \
             patch.object(module, 'serial_sample', side_effect=lambda *a: dict(sample, issues=[])), \
             patch.object(module.subprocess, 'Popen', return_value=process):
            _, clean = module.run_serial_monitored(['x'], {}, 'GPU-a', Path(tmp), 'a' * 64)
            self.assertFalse(clean)
            self.assertFalse(json.loads((Path(tmp) / 'serial-telemetry.json').read_text())['clean'])

    def test_busy_target_without_attributable_process_fails(self):
        def sample(util):
            return {'gpu_query_start_ns': 1000000000, 'gpu_query_end_ns': 1000000001,
                    'devices': [{'uuid': 'GPU-a', 'utilization_percent': util}], 'processes': [], 'issues': []}
        process = SimpleNamespace(pid=5, returncode=0, communicate=lambda: ('', ''))
        polls = iter([None, 0, 0])
        process.poll = lambda: next(polls)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(module.os.environ, {'AUTOBENCH_SCHEDULER_JOB_ID': 'job'}), \
             patch.object(module, 'serial_sample', side_effect=[sample(0), sample(90), sample(0)]), \
             patch.object(module.subprocess, 'Popen', return_value=process):
            _, clean = module.run_serial_monitored(['x'], {}, 'GPU-a', Path(tmp), 'a' * 64)
            self.assertFalse(clean)

    def test_dirty_initial_sample_prevents_launch(self):
        sample = {'devices': [{'uuid': 'GPU-a', 'utilization_percent': 0}], 'processes': [], 'issues': ['foreign GPU activity']}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(module.os.environ, {'AUTOBENCH_SCHEDULER_JOB_ID': 'job'}), \
             patch.object(module, 'serial_sample', return_value=sample), \
             patch.object(module.subprocess, 'Popen') as popen:
            _, clean = module.run_serial_monitored(['x'], {}, 'GPU-a', Path(tmp), 'a' * 64)
            self.assertFalse(clean)
            popen.assert_not_called()

    def test_retrospective_activity_requires_successful_bound_child(self):
        import hashlib
        def sample(util):
            return {'gpu_query_start_ns': 2000000000, 'gpu_query_end_ns': 2000000001,
                    'devices': [{'uuid': 'GPU-a', 'utilization_percent': util}], 'processes': [], 'issues': []}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(module.os.environ, {'AUTOBENCH_SCHEDULER_JOB_ID': 'job'}):
            output = Path(tmp) / 'child.json'
            output.write_text('{}')
            def popen(*args, **kwargs):
                path = Path(kwargs['env']['CGS_LIFETIME_EVENTS'])
                start = {'event': 'start', 'invocation_id': 'one', 'pid': 55, 'device_uuid': 'GPU-a',
                         'proc_start_ticks': 123,
                         'binary_sha256': 'a' * 64, 'spawn_lower_ns': 1400000000, 'spawn_upper_ns': 1500000000,
                         'backend': 'direct', 'argv': ['/x/cgs_study', '--mode', 'campaign', '--backend', 'direct', '--output', str(output)]}
                end = {'event': 'end', 'invocation_id': 'one', 'pid': 55, 'proc_start_ticks': 123,
                       'reap_lower_ns': 1799999999, 'reap_upper_ns': 1800000000,
                       'returncode': 0, 'output_path': str(output), 'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
                path.write_text(json.dumps(start) + '\n' + json.dumps(end) + '\n')
                polls = iter([None, 0, 0])
                return SimpleNamespace(pid=5, returncode=0, communicate=lambda: ('', ''), poll=lambda: next(polls))
            with patch.object(module, 'serial_sample', side_effect=[sample(0), sample(90), sample(0)]), \
                 patch.object(module.subprocess, 'Popen', side_effect=popen):
                _, clean = module.run_serial_monitored(['x'], {}, 'GPU-a', Path(tmp), 'a' * 64)
            self.assertTrue(clean)
            evidence = json.loads((Path(tmp) / 'serial-telemetry.json').read_text())
            self.assertEqual(evidence['samples'][1]['attributed_invocation_ids'], ['one'])


if __name__ == '__main__':
    unittest.main()
