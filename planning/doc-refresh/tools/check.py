#!/usr/bin/env python3
"""Offline checks for the prepared package or an adopted local documentation pass.

These checks validate records and presentation contracts, not the truth of a
self-reported benchmark. Native Todo validation and actual run receipts remain
separate evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit
from common import (ROOT,TOKEN,read_json,project_ids,project_dir,safe,
                    substitute,managed_block,validate_result,validate_review)


def plan_check(plan: dict) -> dict:
    if plan.get('schema_version') != 2:raise ValueError('Expected native schema-v2 plan')
    tasks=plan.get('tasks',[])
    ids=[t['id'] for t in tasks]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate task ids')
    if len(tasks)!=5 or sum(t.get('kind')=='epic' for t in tasks)!=1:
        raise ValueError('This bounded pass uses one epic and four outcomes per authority')
    by_id={t['id']:t for t in tasks};graph={k:[] for k in ids}
    for task in tasks:
        parent=task.get('parent_id')
        if parent and parent not in by_id:raise ValueError('Foreign or missing parent')
        if parent:graph[task['id']].append(parent)
        for dep in task.get('depends_on',[]):
            if dep['type']!='task' or dep.get('task_id') not in by_id:
                raise ValueError('Cleanup dependencies must stay in their local authority')
            graph[task['id']].append(dep['task_id'])
        for field in ['exclusive_paths','read_paths','forbidden_paths']:
            for value in task.get('scope',{}).get(field,[]):
                if value=='.':raise ValueError('Repository root is not a native ownership scope')
                safe(ROOT,value)
        for gate in task.get('gates',[]):
            if gate.get('type')!='command' or not gate.get('argv'):raise ValueError('Invalid command gate')
            if gate.get('cwd')=='.' and not gate.get('input_paths'):raise ValueError('Unscoped gate')
            if gate['argv'][1]!='planning/doc-refresh/tools/check.py':raise ValueError('Unexpected gate driver')
            if not (ROOT/'tools/check.py').is_file():raise ValueError('Missing prepared gate script')
    active=set();done=set()
    def visit(key):
        if key in active:raise ValueError('Plan dependency cycle')
        if key in done:return
        active.add(key)
        for dep in graph[key]:visit(dep)
        active.remove(key);done.add(key)
    for key in ids:visit(key)
    return {'tasks':len(tasks),'outcomes':len(tasks)-1,'structural_check':'passed; native validation is separate'}


def configured_check(name: str) -> dict:
    p=project_dir(name);bindings=read_json(p/'inputs/bindings.json')
    bindings={**bindings,'RESULTS_INDEX':'Result index supplied by renderer.'}
    docs=0
    for file in (p/'templates').rglob('*.md'):
        substitute(file.read_text(encoding='utf-8'),bindings);docs+=1
    results=[]
    for file in sorted((p/'inputs/results').glob('*.json')):
        r=read_json(file);validate_result(r)
        if r['project']!=name:raise ValueError('Result belongs to another repository')
        results.append(r['id'])
    moves=read_json(p/'inputs/moves.json')
    clusters=set()
    for move in moves.get('approved_moves',[]):
        for key in ['from','to','reason']:
            if not move.get(key):raise ValueError('A move needs from/to/reason, not only a destination')
        safe(ROOT,move['from']);safe(ROOT,move['to'])
        if move.get('kind')=='source':clusters.add(move.get('cluster',move['from']))
    if len(clusters)>1 and not moves.get('scope_extension_reason'):
        raise ValueError('Multiple source clusters exceed the default small cleanup; record a narrow scope rationale')
    for deletion in moves.get('deletions',[]):
        if not isinstance(deletion,dict) or not deletion.get('reason') or not deletion.get('path'):
            raise ValueError('Deletion needs an explicit reviewed reason and path')
        safe(ROOT,deletion['path'])
    plan=(p/'plan.v2.json') if (p/'plan.v2.json').exists() else ROOT/'plans'/f'{name}.v2.json'
    counts=plan_check(read_json(plan))
    return {'project':name,'templates':docs,'result_records':results,'plan':counts}


def markdown_links(path: Path,root: Path) -> list[str]:
    text=path.read_text(encoding='utf-8')
    text=re.sub(r'```.*?```','',text,flags=re.S)
    errors=[]
    for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        target=raw.strip().split(' "',1)[0].strip('<>')
        bits=urlsplit(target)
        if bits.scheme or bits.netloc:continue
        if not bits.path:continue
        resolved=(path.parent/unquote(bits.path)).resolve()
        if not resolved.is_relative_to(root.resolve()):errors.append(f'{path.name}: link escapes repository: {target}')
        elif not resolved.exists():errors.append(f'{path.relative_to(root)}: missing link {target}')
        if '.todo-orchestrator' in target or 'background-artifacts' in target:
            errors.append(f'{path.name}: public documentation points into private runtime: {target}')
    return errors


def managed_check(project: Path,repo: Path):
    adoption=project/'inputs/adoption.json'
    if not adoption.is_file():raise ValueError('Missing adoption receipt: install the prepared package with the additive installer')
    original=read_json(adoption).get('managed_block_sha256')
    current=managed_block((repo/'AGENTS.md').read_text(encoding='utf-8')) if (repo/'AGENTS.md').exists() else ''
    actual=hashlib.sha256(current.encode()).hexdigest()
    if original!=actual:raise ValueError('Managed Project Control block changed during editorial cleanup')


def handoff_check(value: dict,name: str) -> None:
    if value.get('project')!=name or value.get('program_id')!='DOCS-REFRESH-2026-09':raise ValueError('Wrong handoff authority/program')
    if value.get('status')!='completed' or not value.get('reviewer'):raise ValueError('Handoff is not completed and reviewed')
    if not re.fullmatch('[0-9a-f]{40}',str(value.get('qualified_head',''))):raise ValueError('Handoff needs qualified source HEAD')
    if not value.get('reader_walkthrough') or not value.get('managed_workflow_preserved'):raise ValueError('Missing human-reading or agent-workflow check')
    if value.get('pending'):raise ValueError('Handoff has unresolved items')
    if not value.get('commands'):raise ValueError('Handoff needs actual validation commands/evidence')
    for command in value['commands']:
        if not isinstance(command.get('argv'),list) or not command['argv'] or not command.get('evidence_path'):
            raise ValueError('Command needs exact argv and a portable receipt path')
        if command.get('returncode')!=command.get('expected_exit_code',0):raise ValueError('Unresolved validation failure in handoff')
        if command.get('basis') not in ['executed','verified_existing_receipt']:
            raise ValueError('Command must say executed or verified_existing_receipt')


def local_check(name: str,repo: Path,evidence: bool=False,final: bool=False) -> dict:
    project=project_dir(name);repo=repo.resolve()
    if not repo.is_dir() or not (repo/'.git').exists():raise ValueError('Local check requires an actual Git checkout')
    info=configured_check(name)
    if evidence or final:
        selected=[]
        for file in (project/'inputs/results').glob('*.json'):
            r=read_json(file)
            if r['selection'].get('selected'):validate_result(r,True,repo);selected.append(r['id'])
        review=read_json(project/'inputs/review.json')
        if len(selected)>2:raise ValueError('Select no more than two studies')
        if not selected and not review.get('no_public_result_reason'):raise ValueError('No qualified public evidence or explicit reason for omission')
        info['selected_results']=selected
    if final:
        review=validate_review(project,repo);managed_check(project,repo)
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if head!=review['qualified_head']:raise ValueError('HEAD changed since local document/source qualification; review the delta')
        paths=[repo/f.relative_to(project/'templates') for f in (project/'templates').rglob('*.md')]
        for file in (project/'inputs/results').glob('*.json'):
            r=read_json(file)
            if r['selection'].get('selected'):
                paths.append(repo/'docs/results'/f"{r['id']}.md")
                data=repo/'docs/results/data'/f"{r['id']}.json"
                if not data.is_file() or read_json(data)!=r:raise ValueError('Published result data is absent or stale')
                if r.get('chart')!='table' and not (repo/'docs/results/assets'/f"{r['id']}.png").is_file():raise ValueError('Selected result figure missing')
        failures=[]
        for path in paths:
            if not path.is_file():failures.append(f'Missing public file: {path}');continue
            if TOKEN.search(path.read_text(encoding='utf-8')):failures.append(f'Unresolved binding: {path}')
            failures.extend(markdown_links(path,repo))
        if failures:raise ValueError('\n'.join(failures))
        handoff=read_json(project/'inputs/handoff.json');handoff_check(handoff,name)
        if handoff['qualified_head']!=head:raise ValueError('Handoff source identity differs from reviewed HEAD')
        if set(handoff['result_ids'])!=set(info['selected_results']):raise ValueError('Handoff result selection differs')
        for command in handoff['commands']:
            if not safe(repo,command['evidence_path']).is_file():raise ValueError('Validation command receipt is unavailable')
        info['final_public_docs_checked']=len(paths)
    return info


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--package',action='store_true');mode.add_argument('--local',action='store_true');mode.add_argument('--final',action='store_true')
    p.add_argument('--project');p.add_argument('--repo',type=Path);p.add_argument('--evidence',action='store_true')
    a=p.parse_args(argv)
    try:
        if a.package:
            records=[configured_check(n) for n in project_ids()]
            common=[(project_dir(n)/'templates/docs/design/program.md').read_bytes() for n in project_ids()]
            if len(set(common))!=1:raise ValueError('Shared program explanation has diverged')
        else:
            if not a.project or a.repo is None:p.error('Local/final requires --project and --repo')
            records=[local_check(a.project,a.repo,a.evidence,a.final)]
        print(json.dumps({'status':'passed','checks':records},indent=2));return 0
    except (ValueError,KeyError,OSError,subprocess.CalledProcessError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)},indent=2));return 1
if __name__=='__main__':raise SystemExit(main())
