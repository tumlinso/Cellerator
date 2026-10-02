#!/usr/bin/env python3
"""Local plan consistency only; NOT the native Todo schema/authority validator."""
import json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    for role,prefix in [('cellerator','CE-MOON-'),('glasshelix','GH-MOON-')]:
        plan=json.loads((root/'machine'/f'{role}.todo-plan.json').read_text())
        assert plan['schema_version']==3
        tasks={x['id']:x for x in plan['tasks']}
        assert len(tasks)==len(plan['tasks']) and all(x.startswith(prefix) for x in tasks)
        graph={x:[d['task_id'] for d in t.get('depends_on',[]) if d['type']=='task'] for x,t in tasks.items()}
        seen=set();visiting=set()
        def walk(x):
            assert x in tasks
            assert x not in visiting,'dependency cycle'
            if x in seen:return
            visiting.add(x)
            for d in graph[x]:walk(d)
            visiting.remove(x);seen.add(x)
        for x in tasks:walk(x)
        for t in tasks.values():
            for ref in t.get('references',[]):
                rel=ref.removeprefix('planning/moonshot-v1/')
                assert (root/rel).is_file(),ref
        for run in plan['runs']:
            for lane in run['lanes']:
                order=lane['tasks'];assert len(order)==len(set(order))
                pos={x:i for i,x in enumerate(order)}
                for x in order:
                    for d in graph[x]:assert pos[d]<pos[x]
        print(f'{role}: {len(tasks)} tasks; IDs, references, DAG and lane order consistent (local only)')

if __name__=='__main__':main()
