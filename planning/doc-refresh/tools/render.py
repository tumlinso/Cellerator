#!/usr/bin/env python3
"""Stage the prewritten documentation. Never overwrite a repository README."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from common import ROOT,project_ids,project_dir,read_json,write_json,substitute,managed_block,validate_result,validate_review,safe,sha

def table(record):
    keys=record['series'];unit=record['metric']['unit']
    lines=['| Case | '+' | '.join(s['label']+' ('+unit+')' for s in keys)+' |', '|---|'+'---:|'*len(keys)]
    for row in record['rows']:
        cells=[]
        for s in keys:
            v=row['values'][s['key']]
            cell=f'{v:.9g}'
            if s['key'] in row.get('bounds',{}):
                lo,hi=row['bounds'][s['key']];cell+=f' ({lo:.9g}–{hi:.9g})'
            cells.append(cell)
        lines.append('| '+row['case']+' | '+' | '.join(cells)+' |')
    return '\n'.join(lines)

def result_page(record,figures):
    state=record['evidence_state'].replace('_',' ')
    parts=['# '+record['title'],f"**Evidence state:** {state}",record['question'],record['claim']]
    if record['rows']:
        if figures and record.get('chart')!='table':parts.append(f"![{record['title']}](assets/{record['id']}.png)")
        parts.append(table(record))
    else:parts.append('**No result is available yet.** The optional fixture and chart structure are prepared, but the actual library must be run before this becomes evidence.')
    parts += ['## What was measured',f"**Scope:** {record['metric']['timing_scope']}. **Statistic:** {record['metric']['statistic']}. **Uncertainty:** {record['metric']['uncertainty']}.",f"**Hardware:** {record.get('hardware','unreported')}"]
    for k,v in record.get('environment',{}).items():parts.append(f"**{k.replace('_',' ').capitalize()}:** {v if v is not None else 'not recovered; verify before stronger claims'}")
    parts += ['## Interpretation and limits','\n'.join('- '+s for s in record['limitations']),'## Reproduce / inspect',record.get('reproduce','')]
    source=record['source']
    if source.get('path'):parts.append(f"[Original recorded explanation](../../{source['path']}) · observed {source.get('observed_at','date unrecorded')} at `{source.get('repository_head_when_read','source pending')}`.")
    if source.get('measurement_revision'):parts.append(f"Measurement source: `{source['measurement_revision']}` (different from the current document-read revision where stated).")
    if source.get('file_identity'):parts.append('Observed source-file identity: `'+source['file_identity']+'`.')
    evidence=record['review'].get('public_evidence_paths',[])
    if evidence:parts.append('Portable evidence: '+', '.join(f'[{Path(p).name}](../../{p})' for p in evidence)+'.')
    elif record['evidence_state']!='pending':parts.append('**Before public promotion:** verify/export portable raw evidence; this preparation preview is based on an inspected summary.')
    parts.append(f"[Chart/table input](data/{record['id']}.json) · [All selected results](index.md)")
    return '\n\n'.join(parts)+'\n'

def stage(name: str,out: Path,repo: Path|None=None,final: bool=False,figures: bool=False):
    project=project_dir(name)
    if (out/'.git').exists() or (repo is not None and out.resolve()==repo.resolve()):
        raise ValueError('Stage to a separate directory, not the repository root')
    if final:validate_review(project,repo)
    bindings=read_json(project/'inputs/bindings.json')
    if repo is not None:
        ag=repo/'AGENTS.md';bindings['MANAGED_WORKFLOW_BLOCK']=managed_block(ag.read_text()) if ag.exists() else ''
    records=[]
    for path in sorted((project/'inputs/results').glob('*.json')):
        rec=read_json(path);validate_result(rec,final=final and rec['selection'].get('selected',False),repo=repo)
        if not final or rec['selection'].get('selected'):records.append(rec)
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'.docstage.json'
    previous=read_json(manifest) if manifest.exists() else None
    if previous and previous.get('project')!=name:raise ValueError('Staging directory belongs to another project')
    if not previous and any(out.iterdir()):raise ValueError('Refusing a nonempty unowned staging directory')
    if previous:
        for relative,digest in previous['files'].items():
            path=safe(out,relative)
            if path.exists() and sha(path)!=digest:raise ValueError(f'Staged file was edited; preserve it before rebuilding: {relative}')
        for relative in previous['files']:
            path=safe(out,relative)
            if path.is_file():path.unlink()
    result_index=[]
    for rec in records:
        suffix=' — optional, not yet executed' if rec['evidence_state']=='pending' else (' — recorded summary pending local verification' if rec['evidence_state']=='recorded_summary' else '')
        result_index.append(f"- [{rec['title']}]({rec['id']}.md){suffix}. {rec['claim']}")
        dest=out/'docs/results';dest.mkdir(parents=True,exist_ok=True)
        (dest/f"{rec['id']}.md").write_text(result_page(rec,figures),encoding='utf-8')
        write_json(dest/'data'/f"{rec['id']}.json",rec)
        if figures:
            from figures import draw
            draw(rec,dest/'assets')
    bindings['RESULTS_INDEX']='\n'.join(result_index) if records else 'No public result is qualified yet. The [status snapshot](../status/current.md) states the existing capability and evidence gap.'
    for file in sorted((project/'templates').rglob('*')):
        if not file.is_file():continue
        dest=safe(out,file.relative_to(project/'templates').as_posix());dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(substitute(file.read_text(encoding='utf-8'),bindings),encoding='utf-8')
    tracked={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='.docstage.json'}
    write_json(manifest,{'project':name,'mode':'final' if final else 'draft','files':tracked})
    return {'project':name,'out':str(out),'documents':len(list(out.rglob('*.md'))),'results':len(records),'mode':'final' if final else 'draft'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',default='all');p.add_argument('--out',type=Path,required=True);p.add_argument('--repo',type=Path);p.add_argument('--final',action='store_true');p.add_argument('--figures',action='store_true')
    a=p.parse_args();names=project_ids() if a.project=='all' else [a.project]
    if a.repo and len(names)!=1:p.error('--repo requires exactly one project')
    print(json.dumps([stage(n,a.out/n if len(names)>1 or a.project=='all' else a.out,a.repo,a.final,a.figures) for n in names],indent=2))
if __name__=='__main__':main()
