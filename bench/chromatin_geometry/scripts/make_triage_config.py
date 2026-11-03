#!/usr/bin/env python3
"""Write the deterministic small full-chain correctness triage matrix."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
parser.add_argument('--replay', help='relative path within the extracted replay closure')
parser.add_argument('--replay-sha256')
args = parser.parse_args()
if bool(args.replay) != bool(args.replay_sha256):
    parser.error('--replay and --replay-sha256 must be supplied together')
case = 'dag2304-64root-complete-consumer-diverse-dense-low-update-r1-change100'
runs = []
for gate in ('supplied', 'ste', 'teacher'):
    update = 'none' if gate == 'ste' else 'incremental'
    for body, backends in (('product', ('scalar', 'direct')),
                           ('linear', ('scalar', 'direct', 'library', 'library-batched'))):
        for backend in backends:
            run = {'case': case, 'gate': gate, 'body': body, 'backend': backend,
                   'precision': 'fp16', 'condition_update': update,
                   'tile': 32, 'stages': 1, 'match_group': f'{gate}-{body}'}
            if args.replay:
                run.update(replay=args.replay, replay_sha256=args.replay_sha256)
            runs.append(run)
target = args.output
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps({'schema': 'cgs.triage.v1', 'runs': runs}, indent=2) + '\n')
print(json.dumps({'output': str(target), 'run_count': len(runs)}))
