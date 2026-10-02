#!/usr/bin/env python3
"""Copy the complete research package to NEW planning directories.

Dry-run by default. Never calls plan apply, edits source, or creates agents.
No cross-repository atomicity is claimed. On an I/O failure, partial new copies
are retained for inspection, never erased along with possible concurrent work.
"""
from __future__ import annotations
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from check_package import verify

ROLES={'cellerator':'Cellerator','glasshelix':'GlassHelix'}
def git(root:Path,*args:str)->str:
    return subprocess.run(['git','-C',str(root),*args],check=True,text=True,capture_output=True).stdout.strip()

def validate_root(raw:str,role:str)->Path:
    p=Path(raw).expanduser().resolve(strict=True)
    actual=Path(git(p,'rev-parse','--show-toplevel')).resolve()
    if p!=actual: raise ValueError(f'{role}: pass the repository root, not a subdirectory: {actual}')
    if not (p/'AGENTS.md').is_file() or not (p/'include'/ROLES[role]).is_dir():
        raise ValueError(f'{role}: expected repository markers missing; refusing to guess')
    return p

def copy_new(src:Path,dst:Path)->None:
    dst.mkdir(exist_ok=False)
    for f in sorted(src.rglob('*')):
        rel=f.relative_to(src)
        if '__pycache__' in rel.parts: continue
        if f.is_symlink(): raise ValueError(f'symlink in package: {f}')
        if f.is_dir(): (dst/rel).mkdir(exist_ok=True)
        elif f.is_file():
            (dst/rel).parent.mkdir(parents=True,exist_ok=True)
            with f.open('rb') as a,(dst/rel).open('xb') as b: shutil.copyfileobj(a,b)

def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cellerator',required=True);ap.add_argument('--glasshelix',required=True)
    ap.add_argument('--apply',action='store_true')
    ap.add_argument('--accept-source-drift',action='store_true',help='Use only after reviewing new source/claims; never skips fresh Todo validation')
    args=ap.parse_args();package=Path(__file__).resolve().parents[1]
    verify(package)
    expected=json.loads((package/'machine/observed-state.json').read_text())['projects']
    roots={role:validate_root(getattr(args,role),role) for role in ROLES}
    a,b=roots.values()
    if a==b or a in b.parents or b in a.parents: raise ValueError('Use independent registered repository roots, not a nested historical gitlink')
    targets=[]
    for role,root in roots.items():
        head=git(root,'rev-parse','HEAD')
        if head!=expected[role]['commit'] and not args.accept_source_drift:
            raise ValueError(f'{role}: source drift {head}; review and explicitly accept before copying')
        parent=root/'planning'
        if parent.is_symlink(): raise ValueError(f'{role}: planning symlink refused')
        dest=parent/'moonshot-v1'
        if dest.exists() or dest.is_symlink(): raise ValueError(f'destination already exists: {dest}; no overwrite')
        dirty=bool(git(root,'status','--porcelain'))
        print(json.dumps({'role':role,'head':head,'dirty':dirty,'destination':str(dest),'action':'copy' if args.apply else 'preview'},sort_keys=True))
        targets.append(dest)
    if not args.apply:
        print('Dry run only. No files, Todo plans or agents changed.');return
    for dest in targets:
        dest.parent.mkdir(exist_ok=True)
        copy_new(package,dest)
        verify(dest)
    print('Copied new planning packages. No Todo import or library-source integration performed.')

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError,json.JSONDecodeError) as e:
        print(f'Installation stopped: {e}',file=sys.stderr);sys.exit(1)
