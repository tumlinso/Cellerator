"""Small offline helpers shared by documentation staging and validation."""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r'\{\{([A-Z][A-Z0-9_]*)\}\}')
MANAGED = re.compile(r'<!-- project-control:start -->.*?<!-- project-control:end -->', re.S)

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def safe(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError(f'Unsafe path: {relative}')
    result=(root/relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes root: {relative}')
    return result

def project_ids():
    if (ROOT/'project.json').is_file():return [read_json(ROOT/'project.json')['id']]
    return sorted(p.name for p in (ROOT/'projects').iterdir() if (p/'project.json').is_file())

def project_dir(name: str) -> Path:
    if not re.fullmatch(r'[a-z][a-z0-9_-]*', name):raise ValueError('Invalid project id')
    if (ROOT/'project.json').is_file():
        if read_json(ROOT/'project.json')['id'] != name:raise ValueError('Wrong local project id')
        return ROOT
    path=safe(ROOT/'projects',name)
    if not (path/'project.json').is_file():raise ValueError(f'Unknown project: {name}')
    return path

def substitute(text: str, bindings: dict) -> str:
    missing=sorted(set(TOKEN.findall(text))-set(bindings))
    if missing:raise ValueError('Unresolved bindings: '+', '.join(missing))
    result=TOKEN.sub(lambda m: str(bindings[m.group(1)]),text)
    if TOKEN.search(result):raise ValueError('Nested/unresolved binding')
    return result

def managed_block(text: str) -> str:
    starts=text.count('<!-- project-control:start -->')
    ends=text.count('<!-- project-control:end -->')
    if starts!=ends or starts>1:raise ValueError('Ambiguous managed workflow block')
    match=MANAGED.search(text)
    return match.group(0) if match else ''

def sha(path: Path) -> str:return hashlib.sha256(path.read_bytes()).hexdigest()

def finite_numbers(obj):
    if isinstance(obj, float) and not math.isfinite(obj):raise ValueError('Nonfinite data is not publishable')
    if isinstance(obj, dict):
        for value in obj.values():finite_numbers(value)
    if isinstance(obj,list):
        for value in obj:finite_numbers(value)

def validate_result(record: dict, final: bool=False, repo: Path | None=None):
    finite_numbers(record)
    for key in ['id','project','title','question','evidence_state','source','metric','series','rows','claim','limitations','review','selection']:
        if key not in record:raise ValueError(f'Result missing {key}')
    if not re.fullmatch(r'[a-z0-9-]+',record['id']):raise ValueError('Invalid result id')
    if record['evidence_state'] not in ['recorded_summary','verified_historical','fresh_measurement','pending']:
        raise ValueError('Unknown evidence state')
    keys=[s['key'] for s in record['series']]
    if not keys or len(keys)!=len(set(keys)):raise ValueError('Result series must be unique')
    if record['evidence_state']=='pending' and record['rows']:raise ValueError('Pending result must not contain pretend measurements')
    if record['evidence_state']!='pending' and not record['rows']:raise ValueError('Evidence result has no rows')
    for row in record['rows']:
        if set(row.get('values',{}))!=set(keys):raise ValueError('Missing or extra result series; do not turn missing into zero')
        for key,v in row['values'].items():
            if isinstance(v,bool) or not isinstance(v,(int,float)):raise ValueError('Result value must be numeric')
            if record['metric']['unit'] in ['ns','us','ms','s','probability'] and v<0:raise ValueError('Invalid negative value')
            if record['metric']['unit']=='probability' and v>1:raise ValueError('Invalid probability')
            if key in row.get('bounds',{}):
                lo,hi=row['bounds'][key]
                if not lo<=v<=hi:raise ValueError('Bounds do not include reported value')
    if final:
        if record['evidence_state'] not in ['verified_historical','fresh_measurement']:raise ValueError('Public result needs verified evidence, not draft summary/pending')
        review=record['review']
        if not (review.get('reviewer') and re.fullmatch(r'[0-9a-f]{40}',str(review.get('verified_at_head',''))) and review.get('source_verified') is True and review.get('method_verified') is True):
            raise ValueError('Result review not completed')
        paths=review.get('public_evidence_paths',[])
        if not paths:raise ValueError('Result needs portable public evidence')
        for path in paths:
            if '.todo-orchestrator' in path or 'background-artifacts' in path:raise ValueError('Private runtime is not public evidence')
            if repo is not None and not safe(repo,path).is_file():raise ValueError(f'Missing public evidence: {path}')
        if not record.get('reproduce'):raise ValueError('No reproduction instructions')
    return True

def validate_review(project: Path, repo: Path | None=None):
    value=read_json(project/'inputs/review.json')
    fields=['paths_finalized','status_reconciled','build_instructions_verified','managed_workflow_preserved','source_moves_reviewed','rendered_pages_reviewed','evidence_selection_reviewed']
    if not value.get('reviewer') or not re.fullmatch(r'[0-9a-f]{40}',str(value.get('qualified_head',''))):raise ValueError('Local review must identify reviewer and qualified HEAD')
    if any(value.get(k) is not True for k in fields):raise ValueError('Local document/source/evidence review is incomplete')
    if value.get('pending'):raise ValueError('Resolve the pending review set before publication')
    selected=[]
    for file in sorted((project/'inputs/results').glob('*.json')):
        record=read_json(file)
        if record['selection'].get('selected'):
            validate_result(record,True,repo);selected.append(record)
    if len(selected)>2:raise ValueError('At most two public selected studies per repository')
    # A measured result is desirable, not fabricated when evidence cannot be qualified.
    if not selected and not value.get('no_public_result_reason'):
        raise ValueError('Select evidence or explicitly explain why none can responsibly be published')
    return value
