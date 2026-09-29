#!/usr/bin/env python3
"""Additive planning-package installation; dry run unless --apply is supplied."""
from __future__ import annotations
import argparse
import json
import hashlib
import subprocess
from pathlib import Path
from common import ROOT,project_dir,safe,managed_block,read_json,write_json

def files_for(name):
    project=project_dir(name);out={}
    for relative in ['templates','inputs']:
        for path in (project/relative).rglob('*'):
            if path.is_file():out[path.relative_to(project).as_posix()]=path
    out['project.json']=project/'project.json'
    out['migration-map.md']=project/'migration-map.md'
    out['plan.v2.json']=ROOT/'plans'/f'{name}.v2.json'
    for relative in ['tools','shared','benchmarks']:
        for path in (ROOT/relative).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':out[path.relative_to(ROOT).as_posix()]=path
    out['START_LOCAL.md']=ROOT/'START_LOCAL.md'
    return out

def install(name: str,repo: Path,apply: bool=False):
    repo=repo.resolve()
    if not repo.is_dir() or not (repo/'.git').exists():raise ValueError('Destination must be an existing Git checkout')
    dest=safe(repo,'planning/doc-refresh');files=files_for(name)
    conflicts=[];additions=[]
    for rel,source in files.items():
        if not source.is_file():raise ValueError(f'Package file missing: {source}')
        target=safe(dest,rel)
        if target.exists() and (not target.is_file() or target.read_bytes()!=source.read_bytes()):conflicts.append(rel)
        elif not target.exists():additions.append(rel)
    adoption=dest/'inputs/adoption.json'
    if adoption.exists() and read_json(adoption).get('project')!=name:
        conflicts.append('inputs/adoption.json')
    if conflicts:raise ValueError('Conflicting existing staging files; nothing copied: '+', '.join(conflicts[:10]))
    if apply:
        dest.mkdir(parents=True,exist_ok=True)
        for rel in additions:
            target=safe(dest,rel);target.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive creation: a concurrent file is never overwritten.
            with target.open('xb') as fh:fh.write(files[rel].read_bytes())
    if apply and not adoption.exists():
        original=managed_block((repo/'AGENTS.md').read_text(encoding='utf-8')) if (repo/'AGENTS.md').is_file() else ''
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        write_json(adoption,{'project':name,'head_at_adoption':head,'managed_block_sha256':hashlib.sha256(original.encode()).hexdigest(),'public_docs_changed':False,'todo_changed':False})
    return {'project':name,'destination':str(dest),'files_to_add':len(additions),'applied':apply,'public_docs_changed':False,'todo_changed':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
    print(json.dumps(install(a.project,a.repo,a.apply),indent=2))
if __name__=='__main__':main()
