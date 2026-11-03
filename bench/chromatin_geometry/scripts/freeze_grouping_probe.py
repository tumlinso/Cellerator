#!/usr/bin/env python3
"""Freeze the synthetic grouping-probe contract without revising the v3 study."""
import hashlib
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
plan = repo / 'planning/chromatin-geometry-study-2026-11-02'
study = repo / 'bench/chromatin_geometry'
parent_path = plan / 'contract-freeze-v3.json'
proof_path = study / 'docs/grouping-probe-host-proof-v1.json'
verification_path = study / 'docs/grouping-probe-original-verification-v1.json'
output = plan / 'contract-freeze-grouping-probe-v4.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
parent, proof, verification = (json.loads(p.read_text()) for p in (parent_path, proof_path, verification_path))
if output.exists():
    raise RuntimeError('refusing to replace an existing frozen contract')
if sha(parent_path) != '4033eea4a8168f5e898212873252c8782943d12f5325e5a31ca45487f9751f06':
    raise RuntimeError('v3 parent identity differs')
if (proof.get('status') != 'pass' or proof.get('original_contract_sha256') != sha(parent_path) or
        proof.get('recomputed_verification_sha256') != sha(verification_path) or
        proof.get('original_workloads_compared') != 10 or not proof.get('all_original_workloads_equal') or
        not proof.get('original_numerical_bounds_equal')):
    raise RuntimeError('original fixture or numerical-bound proof is incomplete')
for key in ('workloads', 'product_bound', 'grouped_linear_bound'):
    if verification.get(key) != parent.get(key):
        raise RuntimeError(f'original frozen {key} changed')
allowed_changes = {'bench/chromatin_geometry/contract.hh', 'bench/chromatin_geometry/contract_smoke.cc',
                   'bench/chromatin_geometry/backends/consumer_fixture.hh',
                   'bench/chromatin_geometry/backends/consumer_tests.cu',
                   'bench/chromatin_geometry/campaign_runner.cu'}
paths = {}
for entry in parent['source_files']:
    name = entry['path']
    if '/data/cache/' in name:
        continue
    path = repo / name
    if not path.is_file():
        raise RuntimeError(f'parent source missing: {name}')
    actual = sha(path)
    if actual != entry['sha256'] and name not in allowed_changes:
        raise RuntimeError(f'unapproved change to parent contract source: {name}')
    paths[name] = {'path': name, 'sha256': actual, 'bytes': path.stat().st_size}
for name, expected in proof['source_sha256'].items():
    path = repo / name
    if sha(path) != expected:
        raise RuntimeError(f'host-proven source has changed: {name}')
    paths[name] = {'path': name, 'sha256': expected, 'bytes': path.stat().st_size}
extra = [
    'planning/chromatin-geometry-study-2026-11-02/contract-freeze-v3.json',
    'planning/chromatin-geometry-study-2026-11-02/bounded-grouping-probe-v1.json',
    'planning/chromatin-geometry-study-2026-11-02/bounded-grouping-resume-2026-11-03.json',
    'bench/chromatin_geometry/docs/grouping-probe-host-proof-v1.json',
    'bench/chromatin_geometry/docs/grouping-probe-original-verification-v1.json',
    'bench/chromatin_geometry/backends/consumers.cu',
    'bench/chromatin_geometry/scripts/probe_packet.py',
    'bench/chromatin_geometry/scripts/probe_budget.py',
    'bench/chromatin_geometry/scripts/register_grouping_probe.py',
    'bench/chromatin_geometry/scripts/campaign.py',
    'bench/chromatin_geometry/scripts/run_packet.py',
    'bench/chromatin_geometry/scripts/kda_driver.py',
    'bench/chromatin_geometry/scripts/freeze_source.py',
    'bench/chromatin_geometry/scripts/freeze_grouping_probe.py',
]
for name in extra:
    path = repo / name
    if not path.is_file():
        raise RuntimeError(f'probe source missing: {name}')
    paths[name] = {'path': name, 'sha256': sha(path), 'bytes': path.stat().st_size}
result = {'format':'cgs.contract-freeze.v1', 'scope':'synthetic grouping-probe-v1',
          'parent_v3_sha256':sha(parent_path), 'product_bound':parent['product_bound'],
          'parent_reference_source_sha256':{x['path']:x['sha256'] for x in parent['source_files'] if '/reference/' in x['path']},
          'original_workloads':parent['workloads'], 'original_workloads_verified':10,
          'host_proof_sha256':sha(proof_path), 'original_verification_sha256':sha(verification_path),
          'probe_content_sha256':proof['probe_content_sha256'], 'probe_cases':proof['probe_cases'],
          'source_files':list(paths.values())}
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'path':str(output), 'sha256':sha(output), 'source_count':len(paths)}))
