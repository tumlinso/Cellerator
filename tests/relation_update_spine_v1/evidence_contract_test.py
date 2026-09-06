#!/usr/bin/env python3
"""Validate bounded RU1 execution receipts, or run adversarial checker tests.

`--receipt FILE --expect FILE` checks against a separately reviewed expectation
manifest. Neither a receipt's passed flag nor its own declared case inventory is
acceptance authority. Artifact paths resolve relative to the receipt directory.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import unittest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def artifact(record, base, expected_hash=None):
    path = base / record['path']
    actual = digest(path.read_bytes())
    require(actual == record['sha256'], 'artifact hash mismatch: ' + str(path))
    if expected_hash is not None:
        require(actual == expected_hash, 'artifact differs from external expectation')
    return path


def raw(record, name):
    text = record[name]
    require(isinstance(text, str) and text, 'missing raw ' + name)
    require(digest(text.encode()) == record[name + '_sha256'], 'raw hash mismatch')
    return text


def sanitizer_summary(text, tool):
    require('COMPUTE-SANITIZER' in text, 'missing sanitizer identity')
    errors = re.findall(r'ERROR SUMMARY:\s*(\d+) errors?', text)
    require(not any(int(value) for value in errors), 'conflicting nonzero error summary')
    if tool == 'racecheck':
        summaries = re.findall(r'RACECHECK SUMMARY:\s*(\d+) hazards?.*?\((\d+) errors?,\s*(\d+) warnings?\)', text)
        require(summaries and all(all(int(n) == 0 for n in row) for row in summaries), 'missing/nonzero racecheck summary')
    else:
        require(errors and all(int(value) == 0 for value in errors), 'missing complete zero-error summary')


def validate(receipt, expected, base):
    require(receipt['kind'] == expected['kind'] == 'device_acceptance', 'source-only/benchmark receipt cannot establish device acceptance')
    require(re.fullmatch(r'[0-9a-f]{40}', receipt['source_commit']) is not None, 'invalid source commit')
    require(receipt['source_commit'] == expected['source_commit'], 'source commit mismatch')
    require(receipt['source_dirty_paths'] == expected['source_dirty_paths'], 'dirty source inventory mismatch')
    for path, sha256 in expected['source_dirty_paths'].items():
        artifact({'path': path, 'sha256': sha256}, base)
    require(receipt['gpu_compute_capability'] == '7.0', 'actual sm70 required')
    require(receipt['gpu_uuid'] == expected['gpu_uuid'] and receipt['gpu_uuid'].startswith('GPU-'), 'device UUID mismatch')
    require(bool(receipt['gpu_name']), 'missing observed device name')
    require(receipt['device_executed'] is True and receipt['skipped'] is False, 'device not executed')
    require(all(receipt['tool_versions'].get(name) for name in ('compiler', 'cuda', 'driver', 'compute_sanitizer')), 'missing tool identities')
    require(receipt['build_argv'] and all(isinstance(x, str) for x in receipt['build_argv']), 'missing build command')
    require(receipt['build_config'] == expected['build_config'], 'build configuration mismatch')
    lease_path = artifact(receipt['controller_lease'], base)
    lease = json.loads(lease_path.read_text())
    require(lease['format'] == 'CUDA-FOREGROUND-LEASE/1' and 'accelerator:' + receipt['gpu_uuid'] in lease['resource_ids'], 'missing controller device lease evidence')
    mutex_path = artifact(receipt['controller_log'], base)
    mutex_text = mutex_path.read_text()
    require('[benchmark-mutex] acquired' in mutex_text and '[benchmark-mutex] released' in mutex_text, 'missing actual mutex acquisition/release log')
    wanted = {(r['test'], r['tool']): r for r in expected['required_runs']}
    require(wanted and len(wanted) == len(expected['required_runs']), 'empty/duplicate external expected inventory')
    runs = receipt['runs']
    require(len(runs) == len(wanted), 'partial or excess execution inventory')
    seen = set()
    for run in runs:
        key = (run['test'], run['tool'])
        require(key in wanted and key not in seen, 'unknown/duplicate execution')
        seen.add(key)
        want = wanted[key]
        require(run['status'] == 'completed' and run['executed'] is True and run['skipped'] is False, 'skipped/timeout/source-only execution')
        require(type(run['exit_code']) is int and run['exit_code'] == 0, 'nonzero/invalid exit')
        require(run['fixture_identity'] == want['fixture_identity'] and run['numerical_policy'] == want['numerical_policy'], 'fixture/profile mismatch')
        binary = artifact(run['binary'], base, want['binary_sha256'])
        require(run['argv'] == want['argv'], 'run command differs from external expectation')
        require(run['argv'] and str(binary.resolve()) in run['argv'], 'argv does not name tested binary')
        require(run['tool'] in ('memcheck', 'racecheck', 'synccheck'), 'unsupported sanitizer claim')
        require('--tool' in run['argv'] and run['argv'][run['argv'].index('--tool') + 1] == run['tool'], 'argv/tool mismatch')
        text = raw(run, 'log')
        for name in ('stdout', 'stderr'):
            require(digest(run[name].encode()) == run[name + '_sha256'], 'output hash mismatch')
        require(not re.search(r'(?im)^\s*(?:SKIPPED|NO GPU|TIMEOUT)\b', text + run['stdout'] + run['stderr']), 'raw output reports skipped execution')
        sanitizer_summary(text, run['tool'])
    require(seen == set(wanted), 'missing execution')
    return True


def validate_samples(samples):
    require(samples, 'no benchmark samples')
    groups = {}
    fixture_identities = {}
    for row in samples:
        require(row['correct'] is True, 'benchmark numerical failure')
        require(row['reset'] == 'logical_f16_upload_outside_timing', 'missing reset outside timing')
        require(row['fixture_fnv1a64'] and row['seed'], 'missing fixture provenance')
        require(row['profile'] in ('half', 'full_f32'), 'unknown numerical policy')
        require(row['width'] in (1, 16) and (row['width'] != 1 or row['route'] == 'sparse'), 'unsupported width/route')
        require(row['route'] in ('sparse', 'hybrid') and (row['route'] != 'hybrid' or row['profile'] == 'half'), 'ineligible route/profile')
        identity_key = (row['fixture'], row['modules'], row['width'])
        identity = (row['fixture_fnv1a64'], row['seed'])
        require(fixture_identities.setdefault(identity_key, identity) == identity, 'compared routes/profiles used different operands')
        require(row['horizon'] in (1, 16, 128, 1024), 'undeclared reuse horizon')
        for name in ('topology_prepare_ms', 'gradient_prepare_ms', 'resident_gpu_ms', 'resident_wall_ms', 'reset_ms', 'h2d_ms', 'observation_d2d_d2h_ms'):
            require(type(row[name]) in (float, int) and math.isfinite(row[name]) and row[name] >= 0, 'invalid timing')
        require(row['resident_gpu_ms'] > 0 and row['resident_wall_ms'] > 0, 'unmeasured resident timing')
        key = tuple(row[x] for x in ('fixture_fnv1a64', 'route', 'profile', 'width', 'horizon', 'operand_reuse', 'diagnostic', 'profiled'))
        group = groups.setdefault(key, [])
        require(row['sample'] not in [r['sample'] for r in group], 'duplicate sample')
        group.append(row)
        if row['width'] == 16:
            require(row['updates'] == row['gradient_launches'] == row['horizon'], 'missing actual mutation')
            if row['route'] == 'hybrid':
                require(row['wmma_launches'] > 0, 'missing actual WMMA')
                if row['fixture'] == 'mixed':
                    require(row['residual_launches'] > 0, 'missing mixed residual')
    for rows in groups.values():
        require(len(rows) == rows[0]['repeats'] and {r['sample'] for r in rows} == set(range(rows[0]['repeats'])), 'partial benchmark sample set')
    return True


class TamperTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        (self.base / 'binary').write_bytes(b'actual-test-binary-fixture')
        lease = json.dumps({'format': 'CUDA-FOREGROUND-LEASE/1', 'resource_ids': ['accelerator:GPU-test']}).encode()
        (self.base / 'lease.json').write_bytes(lease)
        mutex = b'[benchmark-mutex] acquired cuda-controller-foreground\n[benchmark-mutex] released cuda-controller-foreground\n'
        (self.base / 'mutex.log').write_bytes(mutex)
        binary_sha = digest((self.base / 'binary').read_bytes())
        self.expected = {'kind': 'device_acceptance', 'source_commit': 'a' * 40, 'source_dirty_paths': {}, 'gpu_uuid': 'GPU-test', 'build_config': 'Release-sm70', 'required_runs': []}
        self.receipt = dict(self.expected, gpu_compute_capability='7.0', gpu_name='Tesla V100', device_executed=True, skipped=False, tool_versions=dict(compiler='g++12', cuda='12.9', driver='580', compute_sanitizer='12.9'), build_argv=['nvcc', '-arch=sm_70'], controller_lease={'path': 'lease.json', 'sha256': digest(lease)}, controller_log={'path': 'mutex.log', 'sha256': digest(mutex)}, runs=[])
        for tool in ('memcheck', 'racecheck', 'synccheck'):
            want = dict(test='numerics', tool=tool, binary_sha256=binary_sha, fixture_identity='dense-v1', numerical_policy='half', argv=['compute-sanitizer', '--tool', tool, str(self.base / 'binary')])
            self.expected['required_runs'].append(want)
            log = '========= COMPUTE-SANITIZER\n' + ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n' if tool == 'racecheck' else '========= ERROR SUMMARY: 0 errors\n')
            self.receipt['runs'].append(dict(want, status='completed', executed=True, skipped=False, exit_code=0, binary={'path': 'binary', 'sha256': binary_sha}, log=log, log_sha256=digest(log.encode()), stdout='checks PASS\n', stdout_sha256=digest(b'checks PASS\n'), stderr='', stderr_sha256=digest(b'')))

    def reject(self, mutation):
        candidate = copy.deepcopy(self.receipt)
        mutation(candidate)
        with self.assertRaises((ValueError, KeyError, OSError, IndexError)):
            validate(candidate, self.expected, self.base)

    def test_valid(self):
        self.assertTrue(validate(self.receipt, self.expected, self.base))

    def test_false_passes(self):
        for field, value in [('kind', 'source_only'), ('device_executed', False), ('skipped', True), ('gpu_compute_capability', '8.0'), ('source_commit', 'b' * 40), ('gpu_uuid', 'GPU-other')]:
            with self.subTest(field=field):
                self.reject(lambda r: r.update({field: value, 'passed': True}))

    def test_inventory(self):
        self.reject(lambda r: r['runs'].pop())
        self.reject(lambda r: r['runs'].__setitem__(1, copy.deepcopy(r['runs'][0])))
        self.reject(lambda r: r.update(runs=[]))

    def test_run_tamper(self):
        for field, value in [('exit_code', 1), ('exit_code', False), ('status', 'timeout'), ('skipped', True), ('executed', False), ('numerical_policy', 'full_f32'), ('fixture_identity', 'other'), ('log', 'source says WMMA'), ('log_sha256', '0' * 64), ('argv', ['true'])]:
            with self.subTest(field=field):
                self.reject(lambda r: r['runs'][0].update({field: value, 'passed': True}))

    def test_changed_executable(self):
        self.reject(lambda r: r['runs'][0].update(argv=['echo', '--tool', 'memcheck', str(self.base / 'binary')]))

    def test_conflicting_and_missing_summaries(self):
        for text in ['COMPUTE-SANITIZER\n', 'COMPUTE-SANITIZER\nERROR SUMMARY: 0 errors\nERROR SUMMARY: 1 errors\n', 'COMPUTE-SANITIZER\nERROR SUMMARY: 0 errors\nSKIPPED no GPU\n']:
            self.reject(lambda r: r['runs'][0].update(log=text, log_sha256=digest(text.encode())))
        text = 'COMPUTE-SANITIZER\nRACECHECK SUMMARY: 0 hazards displayed (0 errors, 1 warnings)\n'
        self.reject(lambda r: r['runs'][1].update(log=text, log_sha256=digest(text.encode())))

    def test_changed_and_missing_artifacts(self):
        (self.base / 'binary').write_bytes(b'tampered')
        with self.assertRaises(ValueError): validate(self.receipt, self.expected, self.base)
        (self.base / 'binary').unlink()
        with self.assertRaises(OSError): validate(self.receipt, self.expected, self.base)

    def test_lease(self):
        (self.base / 'lease.json').write_text('{}')
        with self.assertRaises(ValueError): validate(self.receipt, self.expected, self.base)

    def test_dirty_source_hash(self):
        (self.base / 'dirty.cc').write_text('reviewed source')
        hashes = {'dirty.cc': digest(b'reviewed source')}
        self.expected['source_dirty_paths'] = hashes
        self.receipt['source_dirty_paths'] = hashes
        self.assertTrue(validate(self.receipt, self.expected, self.base))
        (self.base / 'dirty.cc').write_text('changed source')
        with self.assertRaises(ValueError): validate(self.receipt, self.expected, self.base)

    def test_sample_controls(self):
        row = dict(correct=True, reset='logical_f16_upload_outside_timing', fixture_fnv1a64='abc', seed='formula', profile='half', horizon=16,
                   fixture='mixed', modules=1, route='hybrid', width=16, operand_reuse='refresh', diagnostic=False, profiled=False, sample=0, repeats=1,
                   updates=16, gradient_launches=16, wmma_launches=16, residual_launches=16)
        for key in ('topology_prepare_ms', 'gradient_prepare_ms', 'resident_gpu_ms', 'resident_wall_ms', 'reset_ms', 'h2d_ms', 'observation_d2d_d2h_ms'):
            row[key] = 1.0
        self.assertTrue(validate_samples([row]))
        for field, value in [('reset', 'no_reset'), ('repeats', 2), ('correct', False), ('wmma_launches', 0), ('residual_launches', 0), ('updates', 15), ('width', 32), ('width', 1), ('resident_gpu_ms', float('nan'))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_samples([dict(row, **{field: value})])
        with self.assertRaises(ValueError): validate_samples([dict(row, fixture='irregular', wmma_launches=0)])
        with self.assertRaises(ValueError): validate_samples([row, row])
        with self.assertRaises(ValueError): validate_samples([row, dict(row, route='sparse', fixture_fnv1a64='changed')])

    def test_nonfinite_timing(self):
        row = dict(correct=True, reset='logical_f16_upload_outside_timing', fixture_fnv1a64='abc', seed='formula', profile='half', horizon=1, route='sparse', fixture='dense', modules=1, width=16)
        for value in (float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError): validate_samples([dict(row, topology_prepare_ms=value)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--expect', type=Path)
    parser.add_argument('--samples', type=Path)
    args = parser.parse_args()
    if args.samples:
        validate_samples([json.loads(line) for line in args.samples.read_text().splitlines() if line.startswith('{')])
        print('benchmark sample contract PASS')
    elif args.receipt and args.expect:
        validate(json.loads(args.receipt.read_text()), json.loads(args.expect.read_text()), args.receipt.resolve().parent)
        print('device receipt contract PASS')
    else:
        require(not args.receipt and not args.expect, '--receipt and --expect are both required')
        unittest.main(argv=['evidence_contract_test.py'])


if __name__ == '__main__':
    main()
