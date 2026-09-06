#!/usr/bin/env python3
"""Guarded, owner-run manual bootstrap. No activation, dispatch, worktree or implementation commands.

Offline validation and native preview are non-mutating to Todo. The apply subcommand
is the one intentional authority write and requires an explicit confirmation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import sys
sys.dont_write_bytecode = True
from validate_package import validate

PACKAGE=Path(__file__).resolve().parents[1]
PLAN=PACKAGE/'machine/semantic-spine-v1.todo-plan.json'
BASE='b3340736c8c7b17266bd825c9f079d86e42a5639'
UUID='0ccaac37-dbbf-448e-a5f8-def197a70aba'
CONFIRM='APPLY-CE-SS1-RUN-V1'
LIBRARY={'library/cellerator.cell','library/cellerator/cellerator.ceh'}

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def invoke(argv: list[str],cwd: Path,json_result: bool=False):
    r=subprocess.run(argv,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    if r.returncode:
        raise RuntimeError(f'command failed ({r.returncode}): {argv[0]} {" ".join(argv[1:3])}\n{r.stderr[-6000:]}\n{r.stdout[-6000:]}')
    if json_result:
        try: return json.loads(r.stdout)
        except json.JSONDecodeError as e: raise RuntimeError('tool returned non-JSON output; do not infer success') from e
    return r.stdout.strip()

def check_repo(root: Path,review_head: str,ack_library: bool) -> dict:
    validate(PACKAGE,root,True)
    require = lambda x,msg: None if x else (_ for _ in ()).throw(RuntimeError(msg))
    actual_root=Path(invoke(['git','rev-parse','--show-toplevel'],root)).resolve()
    require(actual_root==root.resolve(),'--source-root must be the Cellerator repository root')
    require(not invoke(['git','status','--porcelain=v1','--untracked-files=normal'],root),
            'repository must be clean; preserve and commit user edits plus delivered files first')
    head=invoke(['git','rev-parse','HEAD'],root)
    require(bool(re.fullmatch(r'[0-9a-f]{40}',review_head or '')) and review_head==head,
            '--review-head must equal the explicit 40-character commit you reviewed')
    invoke(['git','merge-base','--is-ancestor',BASE,'HEAD'],root)
    changed=invoke(['git','diff','--name-only',BASE,'HEAD'],root).splitlines()
    allowed_prefixes=('planning/semantic-spine-v1/','examples/semantic_spine_v1/')
    outside=[p for p in changed if not p.startswith(allowed_prefixes) and p not in LIBRARY]
    require(not outside,'source changed outside package/demo and the two previously dirty library inputs; re-review/rebase the plan, do not force: '+', '.join(outside[:12]))
    require(not (set(changed)&LIBRARY) or ack_library,
            'review the preserved library edits, then explicitly add --acknowledge-preserved-library-edits')
    return {'head':head,'reviewed_changes':changed,
            'preserved_library_hashes':{p:sha(root/p) if (root/p).is_file() else None for p in sorted(LIBRARY)},
            'package_manifest_sha256':sha(PACKAGE/'MANIFEST.sha256'),'plan_file_sha256':sha(PLAN)}

def tool_path(value: str) -> str:
    found=shutil.which(value)
    if not found: raise RuntimeError('project-control executable unavailable; inspect the local installation, do not invent a fallback CLI')
    return str(Path(found).resolve())

def native_preview(root: Path,exe: str) -> dict:
    # The inspected CLI prints JSON automatically and takes no --json flag here.
    data=invoke([exe,'plan','validate','--project','cellerator','--file',str(PLAN)],root,True)
    if not isinstance(data,dict) or data.get('valid') is not True:
        raise RuntimeError('native validator did not return valid=true; no authority write is permitted\n'+json.dumps(data,indent=2)[:6000])
    if data.get('project_uuid')!=UUID or not isinstance(data.get('revision'),int):
        raise RuntimeError('native authority UUID/revision is unavailable or does not match Cellerator')
    pre=data.get('current_observation_preconditions')
    if not isinstance(pre,dict) or not pre.get('workflow_authority_fingerprint'):
        raise RuntimeError('coherent first-class workflow authority is unavailable; never downgrade this plan to schema 2')
    if not isinstance(data.get('plan_digest'),str) or not data['plan_digest']:
        raise RuntimeError('native plan digest unavailable')
    if not isinstance(data.get('would_modify'),list):
        raise RuntimeError('native diff output is incomplete')
    if data.get('would_modify'):
        raise RuntimeError('plan would modify existing records; inspect collisions rather than resetting/reusing them')
    if not isinstance(data.get('would_add'),list):
        raise RuntimeError('native validator/diff output changed; inspect the installed CLI contract')
    return data

def read_receipt(path: Path) -> dict:
    data=json.loads(path.read_text())
    if data.get('kind')!='semantic-spine-native-preview-v1': raise RuntimeError('not a reviewed native preview receipt')
    return data

def save(path: Path,data: dict) -> None:
    path=path.expanduser().resolve()
    # Avoid breaking the integrity manifest or accidentally writing into authority internals.
    if path==PACKAGE or PACKAGE in path.parents or '.todo-orchestrator' in path.parts:
        raise RuntimeError('save runtime receipts outside the package and Todo authority directories')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise RuntimeError('receipt already exists; use a new filename to preserve prior evidence')
    with path.open('x',encoding='utf-8') as f:
        f.write(json.dumps(data,indent=2)+'\n')


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['validate','preview','apply'])
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--project-control',default='project-control')
    ap.add_argument('--review-head',help='Explicit reviewed commit; never a wildcard or force option')
    ap.add_argument('--acknowledge-preserved-library-edits',action='store_true')
    ap.add_argument('--receipt',type=Path,help='Preview output or apply input receipt, outside the package')
    ap.add_argument('--confirm')
    args=ap.parse_args();root=args.source_root.expanduser().resolve()
    if args.action=='validate':
        print(json.dumps(validate(PACKAGE,root,True),indent=2));return 0
    repo=check_repo(root,args.review_head,args.acknowledge_preserved_library_edits)
    exe=tool_path(args.project_control)
    if args.receipt is None: raise RuntimeError('--receipt is required for preview/apply')
    if args.action=='preview':
        result=native_preview(root,exe)
        receipt={'kind':'semantic-spine-native-preview-v1','created_unix':time.time(),'repository':repo,
                 'tool':{'path':exe,'sha256':sha(Path(exe))},'native':result,'authority_to_apply':False}
        save(args.receipt,receipt)
        print(json.dumps({'status':'native_preview_saved','receipt':str(args.receipt),'revision':result['revision'],
                          'would_add':result['would_add'],'would_modify':result['would_modify'],
                          'next':'Review receipt; application is a separate explicit command.'},indent=2));return 0
    if args.confirm!=CONFIRM: raise RuntimeError(f'--confirm must be exactly {CONFIRM}; no mutation was attempted')
    prior=read_receipt(args.receipt.expanduser().resolve())
    if prior['repository']!=repo or prior['tool']!={'path':exe,'sha256':sha(Path(exe))}:
        raise RuntimeError('source, package, demo or executable changed since preview; create a new preview')
    if time.time()-prior['created_unix']>3600 or time.time()<prior['created_unix']:
        raise RuntimeError('preview is stale; review a fresh native preview')
    current=native_preview(root,exe)
    for key in ('project_uuid','revision','plan_digest','would_add','would_modify'):
        if current.get(key)!=prior['native'].get(key):
            raise RuntimeError('native authority changed since reviewed preview: '+key+'; no automatic retry')
    material=('workspace_id','project_uuid','todo_revision','todo_semantic_authority_fingerprint',
              'workflow_revision','workflow_authority_fingerprint','repository_commits','worktrees','run_id')
    previous_pre=prior['native']['current_observation_preconditions']
    current_pre=current['current_observation_preconditions']
    if any(previous_pre.get(k)!=current_pre.get(k) for k in material):
        raise RuntimeError('material observation preconditions changed since preview')
    if not current['would_add']:
        print(json.dumps({'status':'no_changes_to_apply','mutation_performed':False}));return 0
    # Last source check; native service separately checks authority revision under its transaction.
    if check_repo(root,args.review_head,args.acknowledge_preserved_library_edits)!=repo:
        raise RuntimeError('source changed during validation')
    out=args.receipt.expanduser().resolve().with_name(args.receipt.name+'.applied.json')
    if out.exists():
        raise RuntimeError('apply-result receipt already exists; inspect previous application before continuing')
    try:
        result=invoke([exe,'plan','apply','--project','cellerator','--file',str(PLAN)],root,True)
        save(out,{'kind':'semantic-spine-manual-apply-result-v1','created_unix':time.time(),'result':result,
                  'run_activation_attempted':False,'agent_launch_attempted':False})
    except (RuntimeError,OSError) as exc:
        raise RuntimeError('APPLY WAS ATTEMPTED. Inspect current authority before any retry: '+str(exc)) from exc
    print(json.dumps({'status':'manual_apply_returned','result':result,'receipt':str(out),
                      'next':'Inspect the created run/lanes. STOP: no activation, workspaces or agents were started.'},indent=2))
    return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except (RuntimeError,ValueError,KeyError,OSError) as e: raise SystemExit(f'bootstrap STOPPED: {e}')
