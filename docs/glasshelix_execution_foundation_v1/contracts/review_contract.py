#!/usr/bin/env python3
"""Compile and execute bounded contract checks and record governance evidence."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tempfile
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--summary',required=True);p.add_argument('tests',nargs='+');a=p.parse_args()
r=Path.cwd();d=r/'docs/glasshelix_execution_foundation_v1';directory=d/'contracts';results=[]
with tempfile.TemporaryDirectory(prefix='ce-nf1-contract-') as output:
 for name in a.tests:
  source=directory/name;binary=Path(output)/source.stem
  command=['c++','-std=c++20','-Wall','-Wextra','-Werror','-UNDEBUG','-Iinclude',str(source),'src/compute/operation/operation_core_v2/schema.cc','src/execution/program/program_v2.cc','-o',str(binary)]
  for argv in (command,[str(binary)]):
   run=subprocess.run(argv,text=True,capture_output=True);results.append({'argv':argv,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr});print(run.stderr) if run.returncode else None;run.check_returncode()
 head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();header=r/'include/Cellerator/compute/operation/native_foundation_contract.hh'
 review={'task':a.task,'source_base_commit':head,'evidence_level':'executed_host_contract_checks_only','summary':a.summary,'source_hashes':{str(x.relative_to(r)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [header,*directory.glob("*.hh"),*[directory/n for n in a.tests]]},'results':results,'limits':'Host descriptor and policy checks do not qualify numerical execution, external SDK linkage, CUDA, or the complete NF1 toolbox.'}
 artifact=directory/(a.task.lower().replace('ce-nf1-','')+'_review.json');artifact.write_text(json.dumps(review,indent=2)+'\n')
 record={'kind':'nf1-governance-receipt-v1','task_id':a.task,'reviewed':True,'source_commit':head,'reviewer_lane':'CE-NF1-L-C','topics':['source-reconciliation','ownership','acceptance-review',a.task.split('-')[-1]],'artifacts':[{'path':str(artifact.relative_to(r)),'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}]}
 (d/'records'/(a.task.lower()+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(a.task, len(a.tests), 'contract test programs passed')
