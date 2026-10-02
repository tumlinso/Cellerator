#!/usr/bin/env python3
"""Read-only Git/source inventory aid. Does NOT establish live task authority."""
import argparse,hashlib,json,pathlib,re,subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',required=True,type=pathlib.Path)
p.add_argument('--project',required=True,choices=('cellerator','baseplane','glasshelix'))
a=p.parse_args();root=a.repo.resolve(strict=True)
def git(*args):return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True).stdout
head=git('rev-parse','HEAD').decode().strip()
files=git('ls-files','-z').decode().split('\0');rows=[];ids=set()
for name in sorted(filter(None,files)):
    path=root/name
    if path.is_symlink():
        rows.append({'path':name,'kind':'symlink','target':str(path.readlink())});continue
    if not path.is_file():
        rows.append({'path':name,'kind':'missing_or_nonregular'});continue
    # Hash relevant source/metadata, not multi-gigabyte tracked data accidentally.
    relevant=(name.startswith(('include/','src/','planning/','experiments/','tests/','docs/','cmake/','python/','bindings/','components/')) or name in ('README.md','AGENTS.md','CMakeLists.txt'))
    if not relevant:continue
    if path.stat().st_size>4*1024*1024:
        rows.append({'path':name,'kind':'large_file_not_read','bytes':path.stat().st_size});continue
    blob=path.read_bytes();rows.append({'path':name,'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)})
    if path.suffix in ('.json','.md','.yaml','.yml'):
        ids.update(re.findall(r'\b(?:CE|BP|GH|CS|STACK)-[A-Z0-9]+(?:-[A-Z0-9]+)+\b',blob.decode(errors='replace')))
print(json.dumps({'project':a.project,'head':head,'tracked_status':git('status','--porcelain','--untracked-files=no').decode(),
 'files':rows,'textual_task_id_candidates':sorted(ids),'authority':'source_discovery_only','limitation':'Tracked bounded files only. Text IDs may be historical, not tasks. Inspect live authorities, untracked intended work and other worktrees separately.'},indent=2))
