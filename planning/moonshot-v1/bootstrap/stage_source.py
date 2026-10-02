#!/usr/bin/env python3
"""Copy one role's payload to a new claimed experimental directory."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
from check_package import verify
from install import validate_root,copy_new

def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['cellerator','glasshelix'],required=True)
    p.add_argument('--root',required=True);p.add_argument('--apply',action='store_true')
    a=p.parse_args();package=Path(__file__).resolve().parents[1];verify(package)
    root=validate_root(a.root,a.role)
    parent=root/'experiments'
    if parent.is_symlink(): raise ValueError('experiments symlink refused')
    dest=parent/'moonshot-v1'
    if dest.exists() or dest.is_symlink():raise ValueError(f'destination already exists: {dest}')
    print(f'{package/"payload"/a.role} -> {dest}')
    if not a.apply:print('Dry run. Obtain the matching source claim before applying.');return
    parent.mkdir(exist_ok=True);copy_new(package/'payload'/a.role,dest)
    print('Source staged. Public native integration and Todo status remain unchanged.')

if __name__=='__main__':
    try:main()
    except Exception as e:print(f'Staging stopped: {e}',file=sys.stderr);sys.exit(1)
