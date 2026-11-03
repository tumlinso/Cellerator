"""CPU-only tests for the bounded chromatin probe's local GPU ledger."""
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import random
import statistics
import tempfile
import threading
import time
import tarfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/probe_budget.py'
SPEC = importlib.util.spec_from_file_location('cgs_probe_budget_test', SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
PACKET_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/probe_packet.py'
PACKET_SPEC = importlib.util.spec_from_file_location('cgs_probe_packet_budget_integration', PACKET_SCRIPT)
packet = importlib.util.module_from_spec(PACKET_SPEC)
PACKET_SPEC.loader.exec_module(packet)


class ProbeBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'study'
        self.root.mkdir()
        self.lanes = self.root / '.campaign-lanes-20261103'
        self.lane = self.lanes / '0'
        for index in (0, 1, 2):
            data = self.lanes / str(index) / '.autobench/data'
            data.mkdir(parents=True)
            (data / 'closure.tar.gz').write_bytes(b'fixed source archive')
            (data / 'frozen-binary').write_bytes(b'fixed executable')
        self.archive = self.lane / '.autobench/data/closure.tar.gz'
        self.binary = self.lane / '.autobench/data/frozen-binary'
        self.state = self.root / 'runs/probe-budget.json'
        self.ledger = self.root / 'runs/campaign-budget.json'
        self.lock = self.root / 'runs/campaign-budget.lock'
        self.ledger.parent.mkdir(parents=True)
        self.ledger.write_text(json.dumps({
            'schema': 'cgs.budget.v1', 'budget_seconds': 14400, 'reservations': [],
        }))
        self.old = (probe.STUDY, probe.CAMPAIGN, probe.GLOBAL_LEDGER, probe.LOCK)
        probe.STUDY = self.root
        probe.CAMPAIGN = self.root / 'scripts/campaign.py'
        probe.GLOBAL_LEDGER = self.ledger
        probe.LOCK = self.lock
        self.addCleanup(self.restore_globals)
        self.init()

    def restore_globals(self):
        probe.STUDY, probe.CAMPAIGN, probe.GLOBAL_LEDGER, probe.LOCK = self.old
        self.tmp.cleanup()

    def init(self):
        args = SimpleNamespace(
            state=str(self.state), lane_root=str(self.lanes),
            source_sha256=probe.sha(self.archive), binary_sha256=probe.sha(self.binary),
        )
        probe.initialize(args)

    def local(self):
        return json.loads(self.state.read_text())

    def reset_workspace(self):
        self.doCleanups()
        self.setUp()

    def save_global(self, rows):
        self.ledger.write_text(json.dumps({
            'schema': 'cgs.budget.v1', 'budget_seconds': 14400, 'reservations': rows,
        }))

    def add_global(self, state='submitted', phase='qualification', job_id='gpu-job', kind='correctness', lane=0):
        row = {
            'id': str(uuid.uuid4()), 'lane_root': str(self.lanes), 'lane': lane,
            'state': state, 'reserved_seconds': probe.PHASE_LIMITS[phase],
            'charged_seconds': 0 if state in ('cpu_build', 'no_submission') else None,
            'job_id': job_id, 'kind': kind,
            'attempt_id': f'attempt-{job_id}',
            'expected_gpu_resource_id': 'gpu:host:GPU-test',
        }
        with self.ledger.open() as stream:
            current = json.load(stream)['reservations']
        current.append(row)
        self.save_global(current)
        return row

    def lane_registration(self, data, phase='qualification'):
        lane = self.lanes / str(probe.PHASE_LANES[phase])
        kill = probe.PHASE_LIMITS[phase] - probe.CLEANUP_MARGIN
        return lane, {'kill_seconds': kill}, kill, 'd' * 64, 'gpu:host:GPU-test'

    def reconcile_args(self, phase='qualification'):
        return SimpleNamespace(state=str(self.state), phase=phase)

    def controller(self, state='submitted', phase='qualification', job_id='gpu-job', kind='correctness'):
        def call(lane_root, lane_index):
            row = self.add_global(state, phase, job_id, kind, lane=lane_index)
            return 0, json.dumps(row), ''
        return call

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + '\n')
        return path

    def build_adapter(self, lane_index):
        """Retain the lane-local packet build manifest and its original receipts."""
        adapter_dir = self.root / f'runs/builds/lane-{lane_index}'
        adapter_dir.mkdir(parents=True, exist_ok=True)
        archive_sha = probe.sha(self.archive)
        binary_sha = probe.sha(self.binary)
        qualification_binary = adapter_dir / 'cgs_grouping_probe_qualification'
        qualification_binary.write_bytes(b'fixed qualification executable')
        qualification_sha = probe.sha(qualification_binary)
        study_receipt = self.write_json(adapter_dir / 'original-cgs-study-build.json', {
            'schema': 'cgs.build.v1', 'target': 'cgs_study',
            'archive_sha256': archive_sha, 'binary_sha256': binary_sha,
        })
        qualification_receipt = self.write_json(adapter_dir / 'original-qualification-build.json', {
            'schema': 'cgs.build.v1', 'target': 'cgs_grouping_probe_qualification',
            'archive_sha256': archive_sha, 'binary_sha256': qualification_sha,
        })
        study_binary = adapter_dir / 'cgs_study'
        study_binary.write_bytes(self.binary.read_bytes())
        adapter = {
            'schema': 'cgs.grouping-probe-build.v1',
            'archive_sha256': archive_sha,
            'binary_sha256': binary_sha,
            'binary_path': str(study_binary.resolve()),
            'archive_source_path': str(self.archive.resolve()),
            'source_root': 'source',
            'source_manifest_sha256': '1' * 64,
            'cgs_study': {
                'target': 'cgs_study', 'path': 'cgs_study',
                'binary_sha256': binary_sha,
                'original_receipt_path': study_receipt.name,
                'original_receipt_sha256': probe.sha(study_receipt),
            },
            'qualification_binary': {
                'target': 'cgs_grouping_probe_qualification',
                'path': qualification_binary.name,
                'binary_sha256': qualification_sha,
                'original_receipt_path': qualification_receipt.name,
                'original_receipt_sha256': probe.sha(qualification_receipt),
            },
        }
        manifest_path = self.write_json(adapter_dir / 'build.json', adapter)
        return {
            'dir': adapter_dir, 'manifest_path': manifest_path,
            'manifest': adapter, 'study_receipt_sha256': probe.sha(study_receipt),
            'qualification_receipt_sha256': probe.sha(qualification_receipt),
            'qualification_binary_sha256': qualification_sha,
        }

    def runtime_identity(self, result_dir, adapter):
        manifest = adapter['manifest']
        return self.write_json(result_dir / 'runtime-wrapper-identity.json', {
            'schema': 'cgs.runtime-wrapper-identity.v1',
            'snapshot_inputs': {
                '.autobench/data/frozen-build.json': manifest['cgs_study']['original_receipt_sha256'],
                '.autobench/data/qualification-build.json': manifest['qualification_binary']['original_receipt_sha256'],
                '.autobench/data/frozen-binary': manifest['cgs_study']['binary_sha256'],
                '.autobench/data/qualification-binary': manifest['qualification_binary']['binary_sha256'],
            },
        })

    def qualification_proof(self, packet, adapter, scheduler_job, scheduler_attempt):
        campaigns = {}
        for expected in probe.CASE_INPUTS.values():
            case_arms = {}
            for arm_name in ('masked-direct-two-tile', 'cohort-direct-two-tile'):
                raw = packet / 'raw/qualification' / f"{expected['case']}-{arm_name}.json"
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw_record = self.native_campaign_record(expected, arm_name)
                self.write_json(raw, raw_record)
                case_arms[arm_name] = {
                    'raw_path': raw.relative_to(packet).as_posix(), 'raw_sha256': probe.sha(raw),
                }
            campaigns[expected['case']] = {
                'workload_sha256': expected['workload_sha256'],
                'workload_content_sha256': expected['content_sha256'], 'arms': case_arms,
            }
        sanitizers = {}
        checks = []
        for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
            logs = {}
            record = {'kind': 'compute-sanitizer', 'tool': tool, 'status': 'pass'}
            for stream in ('stdout', 'stderr'):
                log = packet / 'raw/qualification' / f'{tool}.{stream}.txt'
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(f'{tool} {stream}: zero errors\n')
                logs[stream] = probe.sha(log)
                record[f'{stream}_path'] = log.relative_to(packet).as_posix()
                record[f'{stream}_sha256'] = logs[stream]
            sanitizers[tool] = {
                'status': 'pass', 'zero_error_summary': True,
                'stdout_sha256': logs['stdout'], 'stderr_sha256': logs['stderr'],
            }
            checks.append(record)
        proof = {
            'schema': 'cgs.grouping-probe-proof.v1', 'status': 'pass',
            'archive_sha256': adapter['manifest']['archive_sha256'],
            'binary_sha256': adapter['manifest']['binary_sha256'],
            'qualification_binary_sha256': adapter['qualification_binary_sha256'],
            'scheduler_job_id': scheduler_job, 'scheduler_attempt_id': scheduler_attempt,
            'device_uuid': 'GPU-test',
            # These are the original CMake receipt hashes, not build.json's hash.
            'cgs_study_build_receipt_sha256': adapter['study_receipt_sha256'],
            'qualification_build_receipt_sha256': adapter['qualification_receipt_sha256'],
            'campaigns': campaigns, 'sanitizers': sanitizers,
            'qualification_checks': checks,
        }
        return self.write_json(packet / 'qualification-proof.json', proof), proof

    @staticmethod
    def native_campaign_record(expected, backend, resident=None, recovery=None, cold=4.0, output_digests=None):
        descriptor = ('exact-gate-planes-no-materialized-workset'
                      if backend == 'masked-direct-two-tile'
                      else 'exact-work-membership-and-raw-group-identity')
        row = {
            'schema': 'cgs.campaign.v1', 'status': 'ok', 'accuracy_accepted': True,
            'validation_scope': 'first-and-last-complete-steps', 'device_uuid': 'GPU-test',
            'case': expected['case'], 'backend': backend, 'gate': 'supplied', 'body': 'product',
            'precision': 'fp16', 'condition_update': 'none', 'input': 'synthetic',
            'reuse_horizon': 1 if '-r1-' in expected['case'] else 16,
            'tile': 32, 'stages': 1, 'cell_tiles_per_cta': 2,
            'workload_sha256': expected['workload_sha256'],
            'workload_content_sha256': expected['content_sha256'],
            'descriptor_validation': descriptor,
            'cell_order': 'contiguous' if backend == 'masked-direct-two-tile' else 'exact-program-cohorts',
            'per_lane_group_mask': backend == 'masked-direct-two-tile',
            'program_sorting': backend == 'cohort-direct-two-tile',
            'numerical_contract': {'input_storage': 'fp16', 'accumulation': 'fp32'},
        }
        if resident is not None:
            row.update(
                resident_complete_horizon_ms=resident,
                recurrent_complete_horizon_ms=resident,
                output_recovery_ms=recovery,
                cold_execution_preparation_ms=cold,
                output_and_vjp_sha256=output_digests,
            )
        return row

    @staticmethod
    def paired_summary(ratios):
        point = statistics.median(ratios)
        rng = random.Random(20261102)
        draws = sorted(statistics.median(rng.choices(ratios, k=6)) for _ in range(10000))

        def percentile(p):
            position = p * (len(draws) - 1)
            low = int(position)
            high = min(low + 1, len(draws) - 1)
            return draws[low] + (draws[high] - draws[low]) * (position - low)

        lower, upper = percentile(.025), percentile(.975)
        practical = point < .95 - 1e-12
        supported = upper < 1.0
        return {
            'paired_ratios_cohort_over_masked': ratios,
            'median_paired_ratio_cohort_over_masked': point,
            'median_paired_latency_reduction': 1.0 - point,
            'paired_bootstrap_95_ratio_interval': [lower, upper],
            'paired_bootstrap_95_latency_reduction_interval': [1.0 - upper, 1.0 - lower],
            'paired_bootstrap_seed': 20261102, 'paired_bootstrap_samples': 10000,
            'practical_latency_reduction_over_5_percent': practical,
            'paired_95_interval_supports_improvement': supported,
            'grouping_beneficial_for_probe': practical and supported,
        }

    def timing_collection(self, packet, phase, proof_sha):
        expected = probe.CASE_INPUTS[phase]
        orders = probe._expected_orders()
        rounds = []
        control_medians = [100, 110, 90, 100, 105, 95]
        candidate_medians = [90, 100, 95, 100, 95, 90]
        for index, order in enumerate(orders):
            arms = {}
            for arm_name in order:
                median = (control_medians[index] if arm_name == 'masked-direct-two-tile'
                          else candidate_medians[index])
                samples = [median - 0.1, float(median), median + 0.1]
                recovery = [0.1, 0.2, 0.3]
                cold = 4.0
                raw_path = packet / 'raw/timing' / f'round-{index:02d}' / f'{arm_name}.json'
                stdout = raw_path.with_suffix('.stdout.txt')
                stderr = raw_path.with_suffix('.stderr.txt')
                output_digests = [f'{(index + 1) * 100 + sample:064x}' for sample in range(3)]
                raw = self.native_campaign_record(
                    expected, arm_name, samples, recovery, cold, output_digests)
                self.write_json(raw_path, raw)
                stdout.parent.mkdir(parents=True, exist_ok=True)
                stdout.write_text('native invocation passed\n')
                stderr.write_text('')
                arms[arm_name] = {
                    'status': 'pass', 'device_uuid': 'GPU-test',
                    'workload_sha256': expected['workload_sha256'],
                    'workload_content_sha256': expected['content_sha256'],
                    'resident_complete_horizon_ms': samples,
                    'resident_median_ms': float(median), 'output_recovery_ms': recovery,
                    'cold_execution_preparation_ms': cold,
                    'raw_path': raw_path.relative_to(packet).as_posix(), 'raw_sha256': probe.sha(raw_path),
                    'stdout_path': stdout.relative_to(packet).as_posix(), 'stdout_sha256': probe.sha(stdout),
                    'stderr_path': stderr.relative_to(packet).as_posix(), 'stderr_sha256': probe.sha(stderr),
                    'output_and_vjp_sha256': output_digests,
                    'descriptor_validation': raw['descriptor_validation'],
                }
            ratio = (arms['cohort-direct-two-tile']['resident_median_ms'] /
                     arms['masked-direct-two-tile']['resident_median_ms'])
            rounds.append({
                'round_id': f'round-{index:02d}', 'order': order, 'arms': arms,
                'status': 'pass', 'paired_ratio_cohort_over_masked': ratio,
            })
        ratios = [row['paired_ratio_cohort_over_masked'] for row in rounds]
        collection = {
            'schema': 'cgs.grouping-probe-timing.v1', 'status': 'pass',
            'case': expected['case'], 'workload_sha256': expected['workload_sha256'],
            'workload_content_sha256': expected['content_sha256'],
            'qualification_proof_sha256': proof_sha, 'device_uuid': 'GPU-test',
            'warmups_per_invocation': 1, 'measured_sequences_per_invocation': 3,
            'round_order_rule': 'AB,BA,AB,BA,AB,BA', 'rounds': rounds,
            **self.paired_summary(ratios),
        }
        return self.write_json(packet / 'collection.json', collection), collection

    def settlement_fixture(self, phase='r1'):
        """Create a receipt-bound fixture matching probe_packet's nested output."""
        if phase != 'qualification':
            local = self.local()
            qual_adapter = self.build_adapter(0)
            qual_proof = self.root / 'runs/packets/qual/probe-packet/qualification-proof.json'
            qual_packet = qual_proof.parent
            qual_proof, _ = self.qualification_proof(
                qual_packet, qual_adapter, 'gpu-qualification', 'scheduler-attempt-qualification')
            local['jobs'].append({
                'id': 'qualification-passed', 'phase': 'qualification', 'state': 'passed',
                'reserved_seconds': 300, 'charged_seconds': 0,
                'qualification_proof_sha256': probe.sha(qual_proof),
                'qualification_binary_sha256': qual_adapter['qualification_binary_sha256'],
                'build_manifest_sha256': probe.sha(qual_adapter['manifest_path']),
                'cgs_study_build_receipt_sha256': qual_adapter['study_receipt_sha256'],
                'qualification_build_receipt_sha256': qual_adapter['qualification_receipt_sha256'],
            })
            if phase == 'r16':
                local['jobs'].append({
                    'id': 'r1-passed', 'phase': 'r1', 'state': 'passed',
                    'reserved_seconds': 150, 'charged_seconds': 50,
                })
            self.state.write_text(json.dumps(local))

        job_id = f'gpu-{phase}'
        lane_index = probe.PHASE_LANES[phase]
        with patch.object(probe, 'lane_registration', self.lane_registration), \
             patch.object(probe, '_controller_step', self.controller(phase=phase, job_id=job_id)):
            submitted = probe.reconcile(self.reconcile_args(phase))
        row = next(item for item in json.loads(self.ledger.read_text())['reservations']
                   if item['job_id'] == submitted['job_id'])
        attempt = f'scheduler-attempt-{job_id}'
        receipt = self.write_json(self.root / f'runs/receipts/{job_id}.json', {
            'job': {
                'id': job_id, 'state': 'succeeded', 'cleanup_verified': True,
                'attempt': {'attempt_id': attempt},
            },
        })
        packet_parent = self.root / f'runs/packets/{job_id}'
        packet = packet_parent / 'probe-packet'
        packet.mkdir(parents=True)
        adapter = self.build_adapter(lane_index)
        self.runtime_identity(packet_parent, adapter)
        proof_sha = None
        proof_path = None
        if phase == 'qualification':
            proof_path, _ = self.qualification_proof(packet, adapter, job_id, attempt)
            proof_sha = probe.sha(proof_path)
            extra = {'qualification_result': {
                'qualification_proof_path': 'qualification-proof.json',
                'qualification_proof_sha256': proof_sha,
            }}
        else:
            qual_proof_path = self.root / 'runs/packets/qual/probe-packet/qualification-proof.json'
            proof_path = self.write_json(packet / 'qualification-proof.json', json.loads(qual_proof_path.read_text()))
            proof_sha = probe.sha(proof_path)
            collection, _ = self.timing_collection(packet, phase, proof_sha)
            extra = {
                'qualification_proof_path': 'qualification-proof.json',
                'qualification_proof_sha256': proof_sha,
                'collection_path': 'collection.json', 'collection_sha256': probe.sha(collection),
                'serial_telemetry_clean': True,
            }
        metrics = {
            'schema': 'cgs.packet.v1', 'status': 'pass', 'success': True,
            'phase': phase, 'scheduler_job_id': job_id, 'scheduler_attempt_id': attempt,
            'uuid': 'GPU-test', 'archive_sha256': probe.sha(self.archive),
            'binary_sha256': probe.sha(self.binary),
            'binary_path': adapter['manifest']['binary_path'],
            'build_manifest_sha256': probe.sha(adapter['manifest_path']),
            'qualification_binary_sha256': adapter['qualification_binary_sha256'],
            'cgs_study_build_receipt_sha256': adapter['study_receipt_sha256'],
            'qualification_build_receipt_sha256': adapter['qualification_receipt_sha256'],
            **extra,
        }
        self.write_json(packet / 'metrics.json', metrics)
        rows = json.loads(self.ledger.read_text())['reservations']
        row = next(item for item in rows if item['job_id'] == job_id)
        row.update(
            state='settled', charged_seconds=25.0, terminal_state='succeeded',
            worker_status='succeeded', receipt=str(receipt.relative_to(self.root)),
            packet_evidence=str(packet_parent.relative_to(self.root)),
            gpu_resource_id='gpu:host:GPU-test',
            build_evidence=str(adapter['manifest_path'].relative_to(self.root)),
        )
        self.save_global(rows)
        return {
            'job_id': job_id, 'attempt': attempt, 'autobench_attempt': row['attempt_id'],
            'packet': packet, 'packet_parent': packet_parent, 'adapter': adapter,
            'proof_path': proof_path, 'receipt': receipt,
        }

    def source_archive(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        sources = {
            'Cellerator/bench/chromatin_geometry/scripts/run_packet.py': b'# guarded test source\n',
            'Cellerator/bench/chromatin_geometry/scripts/kda_driver.py': b'# tracked test source\n',
        }
        manifest = {'files': {name: {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                              for name, data in sources.items()}}
        members = {**sources, 'SOURCE-MANIFEST.json': json.dumps(manifest).encode()}
        with tarfile.open(path, 'w:gz') as archive:
            for name, content in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(content))
        return path

    def actual_build(self, archive, study_binary, study_receipt, qualifier_binary,
                     qualifier_receipt, build_dir):
        return packet.build(SimpleNamespace(
            archive=archive, archive_sha256=packet.sha256(archive), binary=study_binary,
            receipt=study_receipt, qualification_binary=qualifier_binary,
            qualification_receipt=qualifier_receipt, build=build_dir,
        ))

    def scheduler_result(self, phase, job_id, packet_parent, build_manifest):
        with patch.object(probe, 'lane_registration', self.lane_registration), \
             patch.object(probe, '_controller_step', self.controller(phase=phase, job_id=job_id)):
            submitted = probe.reconcile(self.reconcile_args(phase))
        attempt = f'scheduler-attempt-{job_id}'
        receipt = self.write_json(self.root / f'runs/receipts/{job_id}.json', {
            'job': {'id': job_id, 'state': 'succeeded', 'cleanup_verified': True,
                    'attempt': {'attempt_id': attempt}},
        })
        rows = json.loads(self.ledger.read_text())['reservations']
        row = next(item for item in rows if item['job_id'] == submitted['job_id'])
        row.update(
            state='settled', charged_seconds=25.0, terminal_state='succeeded',
            worker_status='succeeded', receipt=str(receipt.relative_to(self.root)),
            packet_evidence=str(packet_parent.relative_to(self.root)),
            gpu_resource_id='gpu:host:GPU-test',
            build_evidence=str(build_manifest.relative_to(self.root)),
        )
        self.save_global(rows)
        return attempt

    def actual_runtime_identity(self, packet_parent, build_info):
        return self.write_json(packet_parent / 'runtime-wrapper-identity.json', {
            'schema': 'cgs.runtime-wrapper-identity.v1',
            'snapshot_inputs': {
                '.autobench/data/frozen-build.json': build_info['cgs_study']['original_receipt_sha256'],
                '.autobench/data/qualification-build.json': build_info['qualification_binary']['original_receipt_sha256'],
                '.autobench/data/frozen-binary': build_info['cgs_study']['binary_sha256'],
                '.autobench/data/qualification-binary': build_info['qualification_binary']['binary_sha256'],
            },
        })

    def lock_is_free(self):
        with self.lock.open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            finally:
                try:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                except OSError:
                    pass
        return True

    def test_reserves_before_controller_and_releases_lock(self):
        calls = []

        def submit(root, lane):
            calls.append((root, lane))
            local = self.local()
            self.assertEqual(local['jobs'][-1]['state'], 'reserving')
            self.assertEqual(local['jobs'][-1]['reserved_seconds'], 300)
            self.assertTrue(self.lock_is_free(), 'campaign lock held over controller call')
            return self.controller()(root, lane)

        with patch.object(probe, 'lane_registration', self.lane_registration), \
             patch.object(probe, '_controller_step', submit):
            row = probe.reconcile(self.reconcile_args())
        self.assertEqual(row['state'], 'submitted')
        self.assertEqual(len(calls), 1)
        result = probe.status(SimpleNamespace(state=str(self.state)))
        self.assertEqual(result['used_or_reserved_seconds'], 300)
        self.assertEqual(result['gpu_job_count'], 1)

    def test_unknown_controller_outcome_stays_charged_and_blocks_retry(self):
        with patch.object(probe, 'lane_registration', self.lane_registration), \
             patch.object(probe, '_controller_step', return_value=(1, '', 'transport lost')):
            with self.assertRaisesRegex(RuntimeError, 'unknown'):
                probe.reconcile(self.reconcile_args())
            with self.assertRaisesRegex(RuntimeError, 'retries are disabled'):
                probe.reconcile(self.reconcile_args())
        self.assertEqual(probe.status(SimpleNamespace(state=str(self.state)))['used_or_reserved_seconds'], 300)
        self.assertEqual(probe.status(SimpleNamespace(state=str(self.state)))['gpu_job_count'], 1)

    def test_same_phase_and_out_of_order_phase_are_rejected(self):
        with patch.object(probe, 'lane_registration', self.lane_registration), \
             patch.object(probe, '_controller_step', self.controller()):
            with self.assertRaisesRegex(RuntimeError, 'requires the earlier qualification'):
                probe.reconcile(self.reconcile_args('r1'))
            probe.reconcile(self.reconcile_args())
            with self.assertRaisesRegex(RuntimeError, 'retries are disabled'):
                probe.reconcile(self.reconcile_args())

    def test_clear_no_submission_releases_reservation_for_explicit_retry(self):
        calls = []

        def submit(root, lane):
            calls.append(1)
            if len(calls) == 1:
                return self.controller('no_submission', job_id=None, kind=None)(root, lane)
            return self.controller()(root, lane)

        with patch.object(probe, 'lane_registration', self.lane_registration), \
             patch.object(probe, '_controller_step', submit):
            first = probe.reconcile(self.reconcile_args())
            self.assertEqual(first['state'], 'no_submission')
            self.assertEqual(probe.status(SimpleNamespace(state=str(self.state)))['used_or_reserved_seconds'], 0)
            second = probe.reconcile(self.reconcile_args())
        self.assertEqual(second['state'], 'submitted')
        self.assertEqual(len(calls), 2)
        self.assertEqual(probe.status(SimpleNamespace(state=str(self.state)))['gpu_job_count'], 1)

    def test_cpu_build_does_not_consume_gpu_budget_and_requires_clean_receipt_before_retry(self):
        calls = []

        def submit(root, lane):
            calls.append(1)
            if len(calls) == 1:
                return self.controller('cpu_build', job_id='build-1', kind='build')(root, lane)
            return self.controller()(root, lane)

        with patch.object(probe, 'lane_registration', self.lane_registration), \
             patch.object(probe, '_controller_step', submit), \
             patch.object(probe, '_scheduler_job', return_value={'job': {
                 'id': 'build-1', 'state': 'succeeded', 'cleanup_verified': True,
             }}) as scheduler:
            first = probe.reconcile(self.reconcile_args())
            self.assertEqual(first['state'], 'cpu_build')
            self.assertEqual(probe.status(SimpleNamespace(state=str(self.state)))['used_or_reserved_seconds'], 0)
            self.assertEqual(probe.status(SimpleNamespace(state=str(self.state)))['gpu_job_count'], 0)
            second = probe.reconcile(self.reconcile_args())
        self.assertEqual(second['state'], 'submitted')
        scheduler.assert_called_once()
        self.assertEqual(probe.status(SimpleNamespace(state=str(self.state)))['gpu_job_count'], 1)

    def test_unfinished_cpu_build_blocks_retry_without_submission(self):
        calls = []

        def build(root, lane):
            calls.append(1)
            return self.controller('cpu_build', job_id='build-1', kind='build')(root, lane)

        with patch.object(probe, 'lane_registration', self.lane_registration), \
             patch.object(probe, '_controller_step', build), \
             patch.object(probe, '_scheduler_job', return_value={'job': {
                 'id': 'build-1', 'state': 'running', 'cleanup_verified': False,
             }}):
            probe.reconcile(self.reconcile_args())
            with self.assertRaisesRegex(RuntimeError, 'lacks terminal verified cleanup'):
                probe.reconcile(self.reconcile_args())
        self.assertEqual(calls, [1])
        self.assertEqual(probe.status(SimpleNamespace(state=str(self.state)))['gpu_job_count'], 0)

    def test_concurrent_same_phase_calls_reserve_at_most_one_slot(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []
        outcomes = []

        def submit(root, lane):
            calls.append(1)
            entered.set()
            release.wait(2)
            return self.controller()(root, lane)

        def run():
            try:
                probe.reconcile(self.reconcile_args())
                outcomes.append('submitted')
            except RuntimeError as exc:
                outcomes.append(str(exc))

        with patch.object(probe, 'lane_registration', self.lane_registration), \
             patch.object(probe, '_controller_step', submit):
            first = threading.Thread(target=run)
            first.start()
            self.assertTrue(entered.wait(1))
            second = threading.Thread(target=run)
            second.start()
            second.join(1)
            release.set()
            first.join(2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sum(x == 'submitted' for x in outcomes), 1)
        self.assertEqual(probe.status(SimpleNamespace(state=str(self.state)))['gpu_job_count'], 1)

    def test_settlement_uses_actual_qualification_packet_receipts_and_scheduler_attempt(self):
        fixture = self.settlement_fixture('qualification')
        adapter = fixture['adapter']
        metrics = json.loads((fixture['packet'] / 'metrics.json').read_text())
        proof = json.loads(fixture['proof_path'].read_text())
        self.assertTrue(Path(metrics['binary_path']).is_absolute())
        self.assertEqual(metrics['scheduler_attempt_id'], fixture['attempt'])
        self.assertNotEqual(metrics['scheduler_attempt_id'], fixture['autobench_attempt'])
        self.assertEqual(proof['cgs_study_build_receipt_sha256'], adapter['study_receipt_sha256'])
        self.assertEqual(proof['qualification_build_receipt_sha256'], adapter['qualification_receipt_sha256'])
        self.assertNotEqual(proof['cgs_study_build_receipt_sha256'], probe.sha(adapter['manifest_path']))
        self.assertNotEqual(proof['qualification_build_receipt_sha256'], probe.sha(adapter['manifest_path']))
        result = probe.settle(SimpleNamespace(state=str(self.state), phase='qualification'))
        self.assertEqual(result['state'], 'passed', result)

    def test_timing_packet_binds_native_output_and_fixed_paired_summary(self):
        fixture = self.settlement_fixture('r1')
        result = probe.settle(SimpleNamespace(state=str(self.state), phase='r1'))
        self.assertEqual(result['state'], 'passed', result)
        self.assertEqual(result['charged_seconds'], 25.0)
        self.assertEqual(probe.status(SimpleNamespace(state=str(self.state)))['used_or_reserved_seconds'], 25.0)
        timing = json.loads((fixture['packet'] / 'collection.json').read_text())
        self.assertEqual(timing['round_order_rule'], 'AB,BA,AB,BA,AB,BA')
        self.assertEqual([row['order'] for row in timing['rounds']], probe._expected_orders())
        self.assertEqual(timing['paired_ratios_cohort_over_masked'],
                         [row['paired_ratio_cohort_over_masked'] for row in timing['rounds']])
        self.assertNotEqual(
            timing['median_paired_ratio_cohort_over_masked'],
            statistics.median(row['arms']['cohort-direct-two-tile']['resident_median_ms']
                              for row in timing['rounds']) /
            statistics.median(row['arms']['masked-direct-two-tile']['resident_median_ms']
                              for row in timing['rounds']),
        )

    def test_timing_lane_manifests_are_distinct_but_reuse_original_build_identities(self):
        first = self.build_adapter(1)
        second = self.build_adapter(2)
        self.assertNotEqual(probe.sha(first['manifest_path']), probe.sha(second['manifest_path']))
        self.assertNotEqual(first['manifest']['binary_path'], second['manifest']['binary_path'])
        for key in ('archive_sha256', 'binary_sha256'):
            self.assertEqual(first['manifest'][key], second['manifest'][key])
        for key in ('cgs_study', 'qualification_binary'):
            for identity in ('binary_sha256', 'original_receipt_sha256'):
                self.assertEqual(first['manifest'][key][identity], second['manifest'][key][identity])

    def test_actual_packet_producer_output_settles_through_budget_consumer(self):
        archive = self.source_archive(self.root / 'runs/closures/actual-source.tar.gz')
        self.archive = archive
        qualifier_input = self.root / 'runs/inputs/cgs_grouping_probe_qualification'
        qualifier_input.parent.mkdir(parents=True)
        qualifier_input.write_bytes(b'actual producer qualification binary')
        archive_sha = packet.sha256(archive)
        study_sha = packet.sha256(self.binary)
        qualifier_sha = packet.sha256(qualifier_input)
        study_receipt = self.write_json(self.root / 'runs/inputs/cgs-study-receipt.json', {
            'schema': 'cgs.build.v1', 'target': 'cgs_study',
            'archive_sha256': archive_sha, 'binary_sha256': study_sha,
        })
        qualifier_receipt = self.write_json(self.root / 'runs/inputs/qualification-receipt.json', {
            'schema': 'cgs.build.v1', 'target': 'cgs_grouping_probe_qualification',
            'archive_sha256': archive_sha, 'binary_sha256': qualifier_sha,
        })
        for index in (0, 1, 2):
            data_dir = self.lanes / str(index) / '.autobench/data'
            (data_dir / 'closure.tar.gz').write_bytes(archive.read_bytes())
            (data_dir / 'frozen-binary').write_bytes(self.binary.read_bytes())
        local = self.local()
        local['source_archive_sha256'] = archive_sha
        local['binary_sha256'] = study_sha
        self.state.write_text(json.dumps(local))

        qualification_build = self.lanes / '0/.autobench/data/adapter-build'
        qualification_info = self.actual_build(
            archive, self.binary, study_receipt, qualifier_input, qualifier_receipt, qualification_build)
        timing_build = self.lanes / '1/.autobench/data/adapter-build'
        timing_info = self.actual_build(
            archive, self.binary, study_receipt, qualifier_input, qualifier_receipt, timing_build)
        self.assertNotEqual(qualification_info['binary_path'], timing_info['binary_path'])
        for key in ('cgs_study', 'qualification_binary'):
            self.assertEqual(qualification_info[key]['original_receipt_sha256'],
                             timing_info[key]['original_receipt_sha256'])

        qualification_job = 'actual-qualification'
        qualification_attempt = 'scheduler-attempt-actual-qualification'
        qualification_parent = self.root / f'runs/packets/{qualification_job}'
        self.actual_runtime_identity(qualification_parent, qualification_info)
        qualification_args = SimpleNamespace(
            output=str(qualification_parent / 'probe-packet'), uuid='GPU-test',
            build=str(qualification_build), phase='qualification', suite=None,
            qualification_proof=None,
        )

        def fake_native_run(argv, env, text, capture_output, check):
            if Path(argv[0]) == qualification_build / 'cgs_grouping_probe_qualification':
                return packet.subprocess.CompletedProcess(argv, 0, json.dumps({
                    'schema': 'cgs.grouping-probe-qualification.v1', 'status': 'pass',
                    'device_uuid': 'GPU-test',
                }), '')
            if Path(argv[0]) == packet.SANITIZER:
                return packet.subprocess.CompletedProcess(argv, 0, 'ERROR SUMMARY: 0 errors', '')
            case_id = argv[argv.index('--case') + 1]
            backend = argv[argv.index('--backend') + 1]
            raw_path = Path(argv[argv.index('--output') + 1])
            workload = probe.CASE_INPUTS['r1' if '-r1-' in case_id else 'r16']
            self.write_json(raw_path, self.native_campaign_record(workload, backend))
            return packet.subprocess.CompletedProcess(argv, 0, 'native oracle pass', '')

        with patch.dict(os.environ, {
                'AUTOBENCH_SCHEDULER_JOB_ID': qualification_job,
                'AUTOBENCH_SCHEDULER_ATTEMPT_ID': qualification_attempt,
                'CUDA_VISIBLE_DEVICES': 'GPU-test'}), \
             patch.object(packet.subprocess, 'run', side_effect=fake_native_run):
            qualification_metrics = packet.run(qualification_args)
        self.assertTrue(Path(qualification_metrics['binary_path']).is_absolute())
        self.assertNotEqual(qualification_metrics['scheduler_attempt_id'], 'attempt-' + qualification_job)
        qualification_attempt_from_receipt = self.scheduler_result(
            'qualification', qualification_job, qualification_parent,
            qualification_build / 'build.json')
        self.assertEqual(qualification_attempt_from_receipt, qualification_attempt)
        qualification_result = probe.settle(SimpleNamespace(state=str(self.state), phase='qualification'))
        self.assertEqual(qualification_result['state'], 'passed', qualification_result)

        timing_job = 'actual-r1'
        timing_attempt = 'scheduler-attempt-actual-r1'
        timing_parent = self.root / f'runs/packets/{timing_job}'
        self.actual_runtime_identity(timing_parent, timing_info)
        timing_args = SimpleNamespace(
            output=str(timing_parent / 'probe-packet'), uuid='GPU-test',
            build=str(timing_build), phase='r1', suite='kda',
            qualification_proof=str(qualification_parent / 'probe-packet/qualification-proof.json'),
        )
        run_packet_loads = 0

        def fake_tracked(argv, events, lock_path, metadata, raw_path):
            return packet.subprocess.CompletedProcess(argv, 0, 'native run passed', '')

        driver = SimpleNamespace(run_tracked=fake_tracked)

        def fake_run_one(binary, backend, case, warmups, repeats, raw_path):
            self.assertEqual((warmups, repeats), (1, 3))
            base = 1.0 if backend == packet.ARMS[0] else 1.2
            expected = probe.CASE_INPUTS['r1']
            data = self.native_campaign_record(
                expected, backend, resident=[base, base + .1, base - .1],
                recovery=[.2, .2, .2], cold=3.0,
                output_digests=[f'{(int(base * 100) + i):064x}' for i in range(3)])
            self.write_json(raw_path, data)
            argv = packet.campaign_argv(binary, case['id'], backend, raw_path, warmups, repeats)
            driver.run_tracked(argv, os.environ['CGS_LIFETIME_EVENTS'], os.environ['CGS_PROCESS_LOCK'],
                               {'device_uuid': 'GPU-test', 'binary_sha256': study_sha,
                                'backend': backend}, raw_path)
            return data

        driver.run_one = fake_run_one
        serial_module = SimpleNamespace()

        def fake_serial_monitored(command, env, uuid, output, binary_sha):
            child_env = dict(env)
            child_env['CGS_LIFETIME_EVENTS'] = str(self.root / 'runs/native-lifetimes.jsonl')
            child_env['CGS_PROCESS_LOCK'] = str(self.root / 'runs/native-process.lock')
            old_env = dict(os.environ)
            try:
                os.environ.clear()
                os.environ.update(child_env)
                parsed = SimpleNamespace(
                    phase=command[command.index('--phase') + 1],
                    build=command[command.index('--build') + 1],
                    uuid=uuid, output=command[command.index('--output') + 1],
                    qualification_proof=command[command.index('--qualification-proof') + 1],
                )
                packet.collect(parsed)
            finally:
                os.environ.clear()
                os.environ.update(old_env)
            return packet.subprocess.CompletedProcess(command, 0, 'collector passed', ''), True

        serial_module.run_serial_monitored = fake_serial_monitored
        empty_run_packet = SimpleNamespace()

        def load_runtime(name, source_path):
            nonlocal run_packet_loads
            if source_path.name == 'run_packet.py':
                run_packet_loads += 1
                return serial_module if run_packet_loads == 1 else empty_run_packet
            if source_path.name == 'kda_driver.py':
                return driver
            raise AssertionError(f'unexpected runtime module {source_path}')

        with patch.dict(os.environ, {
                'AUTOBENCH_SCHEDULER_JOB_ID': timing_job,
                'AUTOBENCH_SCHEDULER_ATTEMPT_ID': timing_attempt,
                'CUDA_VISIBLE_DEVICES': 'GPU-test'}, clear=False), \
             patch.object(packet, '_load_module', side_effect=load_runtime):
            timing_metrics = packet.run(timing_args)
        self.assertTrue(Path(timing_metrics['binary_path']).is_absolute())
        self.assertNotEqual(timing_metrics['scheduler_attempt_id'], 'attempt-' + timing_job)
        timing_attempt_from_receipt = self.scheduler_result(
            'r1', timing_job, timing_parent, timing_build / 'build.json')
        self.assertEqual(timing_attempt_from_receipt, timing_attempt)
        timing_result = probe.settle(SimpleNamespace(state=str(self.state), phase='r1'))
        self.assertEqual(timing_result['state'], 'passed', timing_result)

    def test_settlement_rejects_altered_original_receipt_or_qualifier_identity(self):
        cases = (
            ('proof original receipt', lambda fixture: self._change_json(
                fixture['proof_path'], 'cgs_study_build_receipt_sha256', '0' * 64), 'receipt'),
            ('qualifier binary identity', lambda fixture: self._change_json(
                fixture['packet'] / 'metrics.json', 'qualification_binary_sha256', '0' * 64), 'manifest'),
            ('source archive identity', lambda fixture: self._change_json(
                fixture['packet'] / 'metrics.json', 'archive_sha256', '0' * 64), 'archive_sha256'),
            ('Scheduler attempt identity', lambda fixture: self._change_json(
                fixture['packet'] / 'metrics.json', 'scheduler_attempt_id', 'autobench-attempt'), 'scheduler_attempt_id'),
        )
        for label, mutate, expected_error in cases:
            with self.subTest(label=label):
                fixture = self.settlement_fixture('qualification')
                mutate(fixture)
                result = probe.settle(SimpleNamespace(state=str(self.state), phase='qualification'))
                self.assertEqual(result['state'], 'failed', result)
                self.assertIn(expected_error, result['failure'])
                self.reset_workspace()

    def test_settlement_rejects_bad_order_paired_statistics_and_native_raw_fields(self):
        cases = (
            ('order', self._tamper_order, 'out of order'),
            ('paired statistics', self._tamper_paired_summary, 'paired summary'),
            ('native raw output', self._tamper_native_raw, 'native output'),
        )
        for label, mutate, expected_error in cases:
            with self.subTest(label=label):
                fixture = self.settlement_fixture('r1')
                mutate(fixture)
                result = probe.settle(SimpleNamespace(state=str(self.state), phase='r1'))
                self.assertEqual(result['state'], 'failed', result)
                self.assertIn(expected_error, result['failure'])
                self.reset_workspace()

    def test_rebind_preserves_the_original_allowance_and_prevents_reset(self):
        parent = self.root / 'runs/grouping-probe-v1/probe-budget.json'
        successor = self.root / 'runs/grouping-probe-v2/probe-budget.json'
        parent_lanes = self.root / 'runs/grouping-probe-v1/lanes'
        successor_lanes = self.root / 'runs/grouping-probe-v2/lanes'
        source_sha, binary_sha = 'a' * 64, 'b' * 64
        probe.initialize(SimpleNamespace(
            state=str(parent), lane_root=str(parent_lanes),
            source_sha256=source_sha, binary_sha256=binary_sha,
        ))
        original_bytes = parent.read_bytes()
        result = probe.rebind(SimpleNamespace(
            parent_state=str(parent), state=str(successor), lane_root=str(successor_lanes),
            source_sha256='c' * 64, binary_sha256='d' * 64,
        ))
        self.assertEqual(parent.read_bytes(), original_bytes)
        self.assertEqual(result['binding_parent'], {
            'path': str(parent), 'sha256': hashlib.sha256(original_bytes).hexdigest(),
            'verified_jobs': 0, 'verified_global_reservations': 0,
        })
        self.assertEqual(result['jobs'], [])
        self.assertEqual(result['preparations'], [])
        self.assertEqual(result['lane_root'], str(successor_lanes))
        self.assertEqual(result['source_archive_sha256'], 'c' * 64)
        self.assertEqual(result['binary_sha256'], 'd' * 64)
        self.assertTrue(parent.with_name(parent.name + '.rebound').is_file())
        with self.assertRaisesRegex(RuntimeError, 'already has a successor'):
            probe.rebind(SimpleNamespace(
                parent_state=str(parent), state=str(successor), lane_root=str(successor_lanes),
                source_sha256='c' * 64, binary_sha256='d' * 64,
            ))
        with self.assertRaisesRegex(RuntimeError, 'existing probe allowance'):
            probe.initialize(SimpleNamespace(
                state=str(self.root / 'runs/probe-budget-reset.json'), lane_root=str(successor_lanes),
                source_sha256='e' * 64, binary_sha256='f' * 64,
            ))

    def test_rebind_rejects_any_prior_job_or_shared_reservation(self):
        parent = self.root / 'runs/grouping-probe-v1/probe-budget.json'
        successor = self.root / 'runs/grouping-probe-v2/probe-budget.json'
        parent_lanes = self.root / 'runs/grouping-probe-v1/lanes'
        successor_lanes = self.root / 'runs/grouping-probe-v2/lanes'
        probe.initialize(SimpleNamespace(
            state=str(parent), lane_root=str(parent_lanes),
            source_sha256='a' * 64, binary_sha256='b' * 64,
        ))
        original = json.loads(parent.read_text())
        original['jobs'].append({'phase': 'qualification', 'state': 'failed'})
        parent.write_text(json.dumps(original))
        args = SimpleNamespace(
            parent_state=str(parent), state=str(successor), lane_root=str(successor_lanes),
            source_sha256='c' * 64, binary_sha256='d' * 64,
        )
        with self.assertRaisesRegex(RuntimeError, 'not provably unused'):
            probe.rebind(args)
        self.assertFalse(successor.exists())
        self.assertFalse(parent.with_name(parent.name + '.rebound').exists())

        self.reset_workspace()
        parent = self.root / 'runs/grouping-probe-v1/probe-budget.json'
        successor = self.root / 'runs/grouping-probe-v2/probe-budget.json'
        parent_lanes = self.root / 'runs/grouping-probe-v1/lanes'
        successor_lanes = self.root / 'runs/grouping-probe-v2/lanes'
        probe.initialize(SimpleNamespace(
            state=str(parent), lane_root=str(parent_lanes),
            source_sha256='a' * 64, binary_sha256='b' * 64,
        ))
        self.save_global([{'lane_root': str(parent_lanes), 'lane': 0,
                           'state': 'failed', 'job_id': 'prior-job'}])
        args = SimpleNamespace(
            parent_state=str(parent), state=str(successor), lane_root=str(successor_lanes),
            source_sha256='c' * 64, binary_sha256='d' * 64,
        )
        with self.assertRaisesRegex(RuntimeError, 'not provably unused'):
            probe.rebind(args)
        self.assertFalse(successor.exists())
        self.assertFalse(parent.with_name(parent.name + '.rebound').exists())

    def _change_json(self, path, key, value):
        data = json.loads(path.read_text())
        data[key] = value
        self.write_json(path, data)
        if path.name == 'metrics.json':
            self._refresh_collection_or_proof_hashes(path.parent)
        elif path.name == 'qualification-proof.json':
            self._refresh_metrics_hash(path.parent)

    def _refresh_metrics_hash(self, packet):
        metrics_path = packet / 'metrics.json'
        metrics = json.loads(metrics_path.read_text())
        if metrics.get('qualification_result'):
            metrics['qualification_result']['qualification_proof_sha256'] = probe.sha(
                packet / metrics['qualification_result']['qualification_proof_path'])
        if metrics.get('qualification_proof_path'):
            metrics['qualification_proof_sha256'] = probe.sha(packet / metrics['qualification_proof_path'])
        self.write_json(metrics_path, metrics)

    def _refresh_collection_or_proof_hashes(self, packet):
        metrics_path = packet / 'metrics.json'
        metrics = json.loads(metrics_path.read_text())
        if metrics.get('collection_path'):
            metrics['collection_sha256'] = probe.sha(packet / metrics['collection_path'])
        if metrics.get('qualification_result'):
            metrics['qualification_result']['qualification_proof_sha256'] = probe.sha(
                packet / metrics['qualification_result']['qualification_proof_path'])
        if metrics.get('qualification_proof_path'):
            metrics['qualification_proof_sha256'] = probe.sha(packet / metrics['qualification_proof_path'])
        self.write_json(metrics_path, metrics)

    def _tamper_order(self, fixture):
        path = fixture['packet'] / 'collection.json'
        data = json.loads(path.read_text())
        data['rounds'][0]['order'].reverse()
        self.write_json(path, data)
        self._refresh_collection_or_proof_hashes(fixture['packet'])

    def _tamper_paired_summary(self, fixture):
        path = fixture['packet'] / 'collection.json'
        data = json.loads(path.read_text())
        data['median_paired_ratio_cohort_over_masked'] += 0.01
        self.write_json(path, data)
        self._refresh_collection_or_proof_hashes(fixture['packet'])

    def _tamper_native_raw(self, fixture):
        path = fixture['packet'] / 'collection.json'
        data = json.loads(path.read_text())
        arm = data['rounds'][0]['arms']['masked-direct-two-tile']
        raw_path = fixture['packet'] / arm['raw_path']
        raw = json.loads(raw_path.read_text())
        raw['resident_complete_horizon_ms'] = [200.0, 201.0, 202.0]
        self.write_json(raw_path, raw)
        arm['raw_sha256'] = probe.sha(raw_path)
        self.write_json(path, data)
        self._refresh_collection_or_proof_hashes(fixture['packet'])

    def test_over_budget_actual_charge_keeps_full_reservation_and_blocks(self):
        with patch.object(probe, 'lane_registration', self.lane_registration), \
             patch.object(probe, '_controller_step', self.controller()):
            submitted = probe.reconcile(self.reconcile_args())
        rows = json.loads(self.ledger.read_text())['reservations']
        row = rows[0]
        job = submitted['job_id']
        receipt = self.root / f'runs/receipts/{job}.json'
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({'job': {
            'id': job, 'state': 'succeeded', 'cleanup_verified': True,
            'attempt': {'attempt_id': 'scheduler-attempt-gpu-job'},
        }}))
        row.update(state='settled', charged_seconds=301, terminal_state='succeeded',
                   receipt=str(receipt.relative_to(self.root)), packet_evidence='runs/packets/missing',
                   gpu_resource_id='gpu:host:GPU-test')
        self.save_global(rows)
        with self.assertRaisesRegex(RuntimeError, 'exceeds the phase budget'):
            probe.settle(SimpleNamespace(state=str(self.state), phase='qualification'))
        self.assertEqual(probe.status(SimpleNamespace(state=str(self.state)))['used_or_reserved_seconds'], 300)
        with self.assertRaisesRegex(RuntimeError, 'retries are disabled'):
            with patch.object(probe, 'lane_registration', self.lane_registration):
                probe.reconcile(self.reconcile_args())


if __name__ == '__main__':
    unittest.main()
