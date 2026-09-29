#!/usr/bin/env python3
"""Read the three local handoffs without merging or writing their authorities."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from common import read_json
from check import handoff_check

def verify(paths: list[Path]):
    values=[read_json(p) for p in paths];names=[v.get('project') for v in values]
    if len(names)!=3 or set(names)!={'cellerator','baseplane','glasshelix'}:raise ValueError('Provide exactly one handoff per project')
    for v in values:handoff_check(v,v['project'])
    return {'status':'passed','program_id':'DOCS-REFRESH-2026-09','authorities_merged':False,
            'projects':[{'project':v['project'],'head':v['qualified_head'],'results':v['result_ids']} for v in values]}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipts',nargs=3,type=Path);a=p.parse_args()
    try:print(json.dumps(verify(a.receipts),indent=2));return 0
    except (ValueError,KeyError,OSError) as e:print(json.dumps({'status':'failed','error':str(e)},indent=2));return 1
if __name__=='__main__':raise SystemExit(main())
