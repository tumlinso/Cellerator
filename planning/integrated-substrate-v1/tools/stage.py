#!/usr/bin/env python3
"""Create-only, inert package staging. Default is dry run; no Todo calls."""
from __future__ import annotations
import argparse,json,pathlib,shutil,subprocess,tempfile
from validate_package import ROOT,validate,verify_manifest

def stage(source:pathlib.Path,repo:pathlib.Path,project:str,apply:bool=False):
    source=source.resolve(strict=True);repo=repo.resolve(strict=True)
    top=pathlib.Path(subprocess.run(['git','-C',str(repo),'rev-parse','--show-toplevel'],check=True,capture_output=True,text=True).stdout.strip()).resolve()
    if top!=repo:raise ValueError('--repo must name the repository root')
    if subprocess.run(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],check=True,capture_output=True,text=True).stdout.strip():
        raise ValueError('tracked source is dirty; do not stage into someone else\'s work')
    validate(source);verify_manifest(source)
    destination=repo/'planning/integrated-substrate-v1'
    if destination.exists() or destination.is_symlink():raise FileExistsError('refusing to overwrite '+str(destination))
    report={'project':project,'destination':str(destination),'mode':'apply' if apply else 'dry_run','imports_plan':False,'activates_tasks':False}
    if not apply:return report
    parent=destination.parent
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):raise ValueError('unsafe planning destination')
    parent.mkdir(parents=True,exist_ok=True)
    temporary=pathlib.Path(tempfile.mkdtemp(prefix='.is1-stage-',dir=parent))
    try:
        manifest=json.loads((source/'MANIFEST.json').read_text())
        for rel in [*manifest['sha256'],'MANIFEST.json']:
            q=temporary/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/rel,q)
        verify_manifest(temporary)
        # mkdir-exclusivity avoids replacing an empty directory created by a racing writer.
        destination.mkdir(exist_ok=False)
        try:
            for child in temporary.iterdir():shutil.move(str(child),str(destination/child.name))
        except Exception:
            # Never delete an already-visible destination blindly; retain failure evidence.
            raise RuntimeError('staging incomplete at '+str(destination)+'; inspect before recovery')
        report['files']=len(manifest['sha256'])+1
    finally:
        shutil.rmtree(temporary,ignore_errors=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True,type=pathlib.Path)
    p.add_argument('--project',required=True,choices=('cellerator','baseplane','glasshelix'));p.add_argument('--apply',action='store_true')
    a=p.parse_args();print(json.dumps(stage(ROOT,a.repo,a.project,a.apply),indent=2))
if __name__=='__main__':main()
