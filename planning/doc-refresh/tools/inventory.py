#!/usr/bin/env python3
"""Read a local tree to classify docs/source. Classification is not move permission."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path

def classify(path: str):
    name=Path(path).name;parts=Path(path).parts
    if '.todo-orchestrator' in parts or name in ['todos.md','todo-status.md'] or (parts and parts[0]=='todos'):return 'generated-operational-state'
    if '.ctxpp' in parts or '.git' in parts:return 'generated-or-tool-state'
    if name=='AGENTS.md':return 'agent-guidance'
    if path=='README.md' or path=='CONTRIBUTING.md':return 'public-entry'
    if 'results' in parts or 'evidence' in parts:return 'evidence-review-needed'
    if 'archive' in parts or 'legacy' in parts:return 'historical-review-needed'
    if parts and parts[0]=='planning':return 'planning-not-architecture'
    if Path(path).suffix in ['.md','.qmd','.rst']:return 'documentation-review-needed'
    if any(p in parts for p in ['experiments','bench','tests','examples']):return 'experiment-test-or-example'
    if Path(path).suffix in ['.cc','.cpp','.cu','.cuh','.hh','.hpp','.h','.c','.py','.cmake']:return 'source-build-or-tool'
    return 'other'

def inventory(repo: Path):
    repo=repo.resolve()
    def git(args):
        result=subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
        return result.stdout.decode('utf-8','replace')
    tracked=[p for p in git(['ls-files','-z']).split('\0') if p]
    untracked=[p for p in git(['ls-files','--others','--exclude-standard','-z']).split('\0') if p]
    tracked_set=set(tracked)
    return {'repository':str(repo),'head':git(['rev-parse','HEAD']).strip(),'status':git(['status','--short']),
            'tracked_count':len(tracked),'untracked_count':len(untracked),
            'files':[{'path':p,'tracked':p in tracked_set,'category':classify(p),'decision':'review'} for p in sorted(set(tracked+untracked))],
            'note':'Complete Git listing at invocation, not proof every file was semantically read. No moves/deletions performed.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=inventory(a.repo);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
