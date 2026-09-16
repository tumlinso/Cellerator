#!/usr/bin/env python3
"""Small source-bound NF1A acceptance runner, not a scientific test generator.

The controller binds real tests and evidence. Structural provenance checks cannot
establish scientific correctness by themselves; independent tests/review remain
part of each outcome's acceptance. Empty or placeholder bindings fail closed.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
sys.dont_write_bytecode = True
from common import PACKAGE, canonical, clean_source, completed_prerequisites, external, git, host_context, load, require, rows, sha, sha_bytes, successful, verify_manifest, write_new


def selected_binding_digest(binding: dict, task: str) -> str:
    return sha_bytes(canonical({**{k:v for k,v in binding.items() if k != 'tasks'}, 'task': binding.get('tasks', {}).get(task)}))

def reference(ref: dict) -> dict:
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str) and isinstance(ref.get('sha256'), str), 'Hashed evidence reference required')
    path = Path(ref['path']); require(path.is_file() and sha(path) == ref['sha256'], 'Evidence hash mismatch: ' + str(path))
    return ref


def junit_ok(path: Path, expected: list[str]) -> None:
    root = ET.parse(path).getroot(); cases = list(root.iter('testcase'))
    require(len(cases) == len(expected) and {c.get('name') for c in cases} == set(expected), 'JUnit did not execute the exact inventory')
    for tag in ('skipped', 'failure', 'error'):
        require(not list(root.iter(tag)), 'JUnit contains ' + tag)
    require(all(c.get('status', 'run') not in ('notrun', 'disabled', 'skipped') for c in cases), 'A test did not run')
    for suite in root.iter('testsuite'):
        require(all(int(suite.get(k, '0')) == 0 for k in ('failures', 'errors', 'disabled', 'skipped')), 'Suite contains failed/skipped tests')


def inventory_ok(data: dict, names: list[str]) -> dict:
    require(names and len(set(names)) == len(names) and all(isinstance(n, str) and n for n in names), 'Nonempty unique exact test names required')
    tests = data.get('tests', []); by = {t['name']: t for t in tests}
    require(len(by) == len(tests) and set(names) <= set(by), 'Duplicate or missing CTest names')
    for name in names:
        test = by[name]; props = {p['name']: p.get('value') for p in test.get('properties', [])}
        require(not props.get('DISABLED') and 'SKIP_RETURN_CODE' not in props and 'SKIP_REGULAR_EXPRESSION' not in props, 'Disabled/skip-enabled required test: ' + name)
        require(test.get('command'), 'CTest command absent')
        require(not any(Path(w).name in ('run_gpu_gate.py', 'run_gate.py') for w in test['command']),
                'Nested outer lease/lock gate is forbidden; bind inner executables/tests instead')
    return {n: by[n] for n in names}


def source_set(spec: dict, roots: dict[str, Path]) -> dict:
    result = {}
    for project, binding in spec.get('sources', {}).items():
        require(project in roots, 'Unregistered source project')
        path = Path(binding['path']).resolve(); observed = clean_source(path)
        a = Path(git(path, 'rev-parse', '--path-format=absolute', '--git-common-dir')).resolve()
        b = Path(git(roots[project], 'rev-parse', '--path-format=absolute', '--git-common-dir')).resolve()
        require(a == b, 'Source path belongs to another repository')
        require(observed['head'] == binding['commit'], 'Source commit differs from bound build input')
        result[project] = {'path': str(path), **observed}
    require(result, 'Actual compiled source set is required')
    return result


def verify_adoption(spec: dict, roots: dict[str, Path], snapshots: dict) -> dict:
    adoption = spec.get('adoption', {}); reviews = adoption.get('import_reviews', {})
    require(set(reviews) == set(roots), 'Both actual import reviews required')
    for project in roots:
        review = load(reference(reviews[project])['path'])
        require(review.get('project') == project, 'Wrong import review')
        require(review['proposal']['observation_preconditions']['project_uuid'] == snapshots[project].project_uuid, 'Wrong imported authority')
        actual = rows(snapshots[project], 'tasks')
        plan = review['proposal']['proposed_change']
        for task in plan['tasks']:
            require(actual.get(task['id'], {}).get('objective') == task['objective'], 'Successor task absent/altered before adoption')
        require(review.get('preserved_successful_task_rows'), 'Missing historical successful-row preservation baseline')
        for tid, expected in review['preserved_successful_task_rows'].items():
            require(tid in actual and sha_bytes(canonical(actual[tid])) == expected, 'Successful historical task altered: ' + tid)
        old_run = ('CE' if project == 'cellerator' else 'GH') + '-NF1-RUN-V1'
        require(rows(snapshots[project], 'workflow_runs').get(old_run, {}).get('status') == 'cancelled', 'Old NF1 run not canonically retired')
        prefix = 'CE-NF1-' if project == 'cellerator' else 'GH-NF1-'
        from common import TERMINAL
        require(all(r.get('status') in TERMINAL for t,r in actual.items() if t.startswith(prefix)), 'Unfinished old NF1 records remain')
    preservation = load(reference(adoption.get('preservation', {}))['path'])
    require(preservation.get('format') == 'nf1a-preservation-v1' and preservation.get('passed') is True, 'Complete byte-preservation receipt required')
    require(sha(preservation['archive']) == preservation['archive_sha256'], 'Preserved dirty-source archive changed')
    dispositions = adoption.get('artifact_dispositions', [])
    require(isinstance(dispositions, list), 'Artifact dispositions required')
    covered = {(d.get('project'), d.get('workspace_id')) for d in dispositions
               if d.get('successor') and d.get('source_reference') and d.get('reason') and d.get('action') in ('preserved_for_successor', 'already_integrated', 'historical_no_source')}
    needed = {(p, w['id']) for p, data in preservation['projects'].items() for w in data['workspace_records']}
    require(needed <= covered, 'Every old NF1 workspace needs an explicit preserved source/ownership disposition')
    for ref in adoption.get('retirement_receipts', []): reference(ref)
    require(len(adoption.get('retirement_receipts', [])) == 2, 'Both canonical retirement receipts required')
    return {'preservation': adoption['preservation'], 'mapped_workspaces': len(covered),
            'meaning': 'Historical artifacts preserved/mapped, not automatically integrated or requalified.'}


def verify_closure(task: str, spec: dict, roots: dict[str, Path]) -> dict:
    refs = spec.get('accepted_receipts', [])
    require(refs, 'Previously passed source-bound acceptance receipts required')
    evidence = [load(reference(r)['path']) for r in refs]
    require(all(r.get('format') == 'nf1a-gate-receipt-v1' and r.get('passed') is True for r in evidence), 'Failed/pending receipt cannot close the program')
    expected = {'CE-NF1A-RELEASE': {'CE-NF1A-QUALIFY', 'GH-NF1A-ACCEPT'},
                'GH-NF1A-RELEASE': {'GH-NF1A-ACCEPT', 'CE-NF1A-RELEASE'},
                'CE-NF1A-CONTROL': {'CE-NF1A-RELEASE'}, 'GH-NF1A-CONTROL': {'GH-NF1A-RELEASE'}}[task]
    require(expected <= {r.get('task_id') for r in evidence}, 'Required source/consumer receipt missing')
    if not task.endswith('-RELEASE'): return {'accepted_receipts': refs}
    sources = source_set(spec, roots); require(set(sources) == set(roots), 'Final source pair required')
    for project, source in sources.items():
        # Final source must be the actual accepted pair; changed code needs requalification.
        supporting = [r['sources'][project] for r in evidence if project in r.get('sources', {})]
        require(supporting and all(s['head'] == source['head'] for s in supporting), 'Final source differs from accepted source; requalify the affected descendant')
        remote = git(roots[project], 'ls-remote', '--exit-code', 'origin', 'refs/heads/main')
        require(remote.split()[0] == source['head'], 'Actual origin/main is not the qualified source for ' + project)
    require(spec.get('remaining_limitations') is not None and spec.get('source_disposition_report'), 'Final limitations and inherited-work disposition report required')
    reference(spec['source_disposition_report'])
    return {'sources': sources, 'accepted_receipts': refs, 'remaining_limitations': spec['remaining_limitations']}


def run_execution(task: str, outcome: dict, spec: dict, common: dict, roots: dict, evidence_dir: Path) -> dict:
    sources = source_set(spec, roots)
    project = outcome['project']; require(project in sources, 'Own source missing')
    require(Path.cwd().resolve() == Path(sources[project]['path']), 'Gate must run in the bound owning worktree')
    if project == 'glasshelix': require('cellerator' in sources, 'GlassHelix must bind the actual Cellerator build source')
    build = Path(spec['build_dir']).resolve(); cache = build / 'CMakeCache.txt'; require(cache.is_file(), 'Configured build required')
    home = re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$', cache.read_text(), re.M)
    require(home and Path(home.group(1)).resolve() == Path.cwd().resolve(), 'Build points to a different source worktree')
    names = spec.get('ctest_names', [])
    inv = subprocess.run(['ctest', '--test-dir', str(build), '--show-only=json-v1'], capture_output=True, text=True, check=True)
    selected = inventory_ok(json.loads(inv.stdout), names)
    coverage = spec.get('coverage', {})
    requirements = {r['id']: r for r in load(PACKAGE/'machine/requirements.json')}
    for rid in outcome['requirements']:
        item = coverage.get(rid, {}); tests = item.get('tests', []); refs = item.get('evidence', [])
        require(set(tests) <= set(names), 'Coverage names a test that will not execute: ' + rid)
        if item.get('disposition') == 'not_selected':
            require(requirements[rid]['classification'] == 'conditional' and item.get('reason') and refs, 'Only genuinely optional work can be unselected with evidence')
        else:
            require(tests or refs, 'Missing substantive acceptance coverage: ' + rid)
        for ref in refs: reference(ref)
    binaries = spec.get('binary_paths', [])
    require(binaries, 'Actual binary/library inventory must be bound, not just an interpreter')
    hashes = {str(Path(p).resolve()): sha(p) for p in binaries}
    for test in selected.values():
        for word in test['command']:
            p = Path(word)
            if p.is_absolute() and p.is_file(): hashes[str(p)] = sha(p)
    cache_hash = sha(cache)
    lease_path = Path(os.environ.get('TODO_GPU_LEASE_RECEIPT', ''))
    require(lease_path.is_file(), 'Native foreground GPU lease required; standalone execution cannot qualify CUDA')
    lease = load(lease_path)
    require(lease.get('format') == 'CUDA-FOREGROUND-LEASE/1' and lease.get('state') == 'active', 'Wrong/inactive native lease')
    require(Path(lease['project_root']).resolve() == Path.cwd().resolve(), 'Lease belongs to another worktree')
    verifier = common.get('gpu_lease_verifier_argv', [])
    require(verifier and all(isinstance(w, str) for w in verifier), 'Bind the inspected live lease verifier')
    verify_cmd = [w.replace('{lease_receipt}', str(lease_path)) for w in verifier]
    lock_path = Path(common['shared_gpu_lock_file']).resolve()
    peer = load(Path(common['peer_bindings']))
    require(Path(peer['shared_gpu_lock_file']).resolve() == lock_path, 'Both projects must use the same host lock')
    evidence = {}
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        checked = subprocess.run(verify_cmd, check=True, capture_output=True, text=True, timeout=60)
        smi = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,compute_cap,driver_version','--format=csv,noheader,nounits'],
                             check=True,capture_output=True,text=True,timeout=30)
        indices = {line.split(',')[0].strip(): line.split(',')[1].strip() for line in smi.stdout.splitlines()}
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        reserved = {r.split('accelerator:',1)[1] for r in lease.get('resource_ids',[]) if r.startswith('accelerator:')}
        actual = [indices.get(t.strip(),t.strip()) for t in visible.split(',') if t.strip()]
        require(reserved and set(actual) == reserved and len(actual) == len(set(actual)), 'Visible GPUs do not match the native lease')
        # Compatibility context for preserved INNER leased test wrappers. The outer
        # lock is owned here; tests must not reacquire it via old outer gate scripts.
        derived = {**common, 'build_dir':str(build),'gpu_lease_receipt':str(lease_path),
                   'gpu_lease_verifier_argv':verifier,'shared_gpu_lock_file':str(lock_path),'evidence_dir':str(evidence_dir)}
        fd, derived_name = tempfile.mkstemp(prefix='nf1a-leased-',suffix='.json',dir=evidence_dir)
        with os.fdopen(fd,'wb') as f: f.write(canonical(derived))
        env = dict(os.environ, NF1_LEASED_BINDINGS=derived_name)
        pattern = '^('+'|'.join(re.escape(n) for n in names)+')$'
        junit = evidence_dir/('junit-'+Path(derived_name).stem+'.xml')
        command = ['ctest','--test-dir',str(build),'-R',pattern,'--no-tests=error','--output-on-failure','--output-junit',str(junit),'-j','1']
        run = subprocess.run(command,env=env,capture_output=True,text=True,timeout=int(spec.get('timeout_seconds',3600)))
        evidence = {'sources':sources,'build_dir':str(build),'binary_sha256':hashes,'cache_sha256':cache_hash,
                    'coverage':coverage,'expected_tests':names,'argv':command,'returncode':run.returncode,
                    'stdout':run.stdout,'stderr':run.stderr,'junit':junit.read_text() if junit.exists() else None,
                    'hardware':{'inventory':smi.stdout,'visible_devices':visible,'lease_sha256':sha(lease_path),'live_verifier':checked.stdout}}
        # Preserve execution output even when a scientific test fails.
        write_new(evidence_dir/(Path(derived_name).stem+'-execution.json'), evidence)
        require(run.returncode == 0, 'Required CTest campaign failed')
        junit_ok(junit,names)
        require(all(sha(p) == digest for p,digest in hashes.items()) and sha(cache)==cache_hash, 'Build artifacts changed during execution')
        after_sources = source_set(spec,roots)
        require({p:(s['path'],s['head'],s['tree']) for p,s in after_sources.items()} == {p:(s['path'],s['head'],s['tree']) for p,s in sources.items()}, 'Source changed during execution')
        subprocess.run(verify_cmd,check=True,capture_output=True,text=True,timeout=60)
    return evidence


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);p.add_argument('--bindings',type=Path,required=True);args=p.parse_args()
    verify_manifest(); outcomes=load(PACKAGE/'machine/outcomes.json'); require(args.task in outcomes,'Unknown outcome')
    outcome=outcomes[args.task];meta=load(PACKAGE/'machine/package.json'); require(outcome['project']==meta['project'],'Wrong local authority')
    binding=load(args.bindings);require(binding.get('pair_id')==meta['pair_id'] and binding.get('project')==meta['project'],'Bindings identity differs')
    spec=binding.get('tasks',{}).get(args.task);require(isinstance(spec,dict) and spec,'Outcome binding is unconfigured')
    _,roots,snapshots=host_context(); prereqs=completed_prerequisites(outcome,snapshots)
    evidence_dir=external(Path(binding['evidence_dir']),list(roots.values()));evidence_dir.mkdir(parents=True,exist_ok=True)
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); path=evidence_dir/(args.task+'-'+stamp+'.json')
    record={'format':'nf1a-gate-receipt-v1','task_id':args.task,'passed':False,'sources':{},'started_at':stamp,
            'binding_sha256':selected_binding_digest(binding,args.task),'prerequisites':prereqs,'meaning':'Source-bound evidence; semantic adequacy still requires the declared tests and review.'}
    # Reserve before device execution. A killed process leaves a pending receipt.
    write_new(path,record)
    try:
        if outcome['gate_kind']=='administrative': record.update(verify_adoption(spec,roots,snapshots))
        elif outcome['gate_kind']=='closure': record.update(verify_closure(args.task,spec,roots))
        else: record.update(run_execution(args.task,outcome,spec,binding,roots,evidence_dir))
        require(selected_binding_digest(load(args.bindings),args.task)==record['binding_sha256'],'Selected outcome/runtime bindings changed during gate')
        _,_,final_snapshots=host_context()
        record['final_prerequisites']=completed_prerequisites(outcome,final_snapshots)
        record['passed']=True
    except Exception as exc:
        record['error']=str(exc)
    finally:
        record['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat()
        with path.open('wb') as out:out.write(canonical(record));out.flush();os.fsync(out.fileno())
    print(json.dumps({'passed':record['passed'],'receipt':str(path),'sha256':sha(path)},sort_keys=True))
    require(record['passed'],record.get('error','Gate failed'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('NF1A gate rejected: '+str(exc),file=sys.stderr);raise SystemExit(1)
