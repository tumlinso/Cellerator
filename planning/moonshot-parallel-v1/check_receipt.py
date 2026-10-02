#!/usr/bin/env python3
"""Read-only, fail-closed CE capability bridge; never runs consumer workloads."""
import argparse,hashlib,json,pathlib,re,subprocess,sys

def require(ok, message):
    if not ok: raise ValueError(message)

def validate_receipt(receipt, ce_root, gh_root, task_done):
    require(receipt.get('schema_version')==1,'unsupported receipt schema')
    require(receipt.get('record_kind')=='moonshot_native_bridge_acceptance','wrong receipt kind')
    require(task_done,'CE-MOON-INTEGRATE is not effectively done')
    source=receipt.get('cellerator_source_commit','')
    require(bool(re.fullmatch('[0-9a-f]{40}',source)),'missing exact CE source identity')
    actual=subprocess.run(['git','-C',str(ce_root),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    require(source==actual,'CE source identity changed; refresh receipt and consumer evidence')
    cap=receipt.get('capability',{})
    require(bool(cap.get('native_call')) and bool(cap.get('framework_call')),'native and framework call paths required')
    require(bool(cap.get('supported_shapes')),'supported shapes required')
    precision=cap.get('precision',{})
    require(all(precision.get(k) for k in ('storage','arithmetic','accumulation','derivative_convention')),'precision/derivative policy missing')
    require(all(cap.get('capabilities',{}).get(k)=='implemented' for k in ('forward','input_vjp','parameter_vjp','jvp')),'required derivative route missing')
    require(isinstance(cap.get('unsupported'),list),'unsupported routes must be explicit')
    checks=receipt.get('checks',[])
    require(bool(checks),'no actual check evidence')
    require({x.get('kind') for x in checks}>={'native_correctness','framework_consumer','derivatives'},'native/framework/derivative checks required')
    for check in checks:
        require(check.get('exit_code')==0 and bool(check.get('argv')),'check command or successful result missing')
        require(check.get('source_commit')==source,'check bound to wrong CE source')
        rel=pathlib.PurePosixPath(check.get('evidence_path',''))
        require(bool(str(rel)) and not rel.is_absolute() and '..' not in rel.parts,'unsafe evidence path')
        root={'cellerator':ce_root,'glasshelix':gh_root}.get(check.get('project'))
        require(root is not None,'unknown evidence project')
        path=root/rel
        require(path.is_file() and not path.is_symlink(),'evidence absent or symlink')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==check.get('sha256'),'evidence hash mismatch')
    require(bool(receipt.get('reviewed_by')) and bool(receipt.get('reviewed_at')),'root acceptance review required')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--receipt',type=pathlib.Path,required=True)
    ap.add_argument('--cellerator',type=pathlib.Path,required=True)
    ap.add_argument('--glasshelix',type=pathlib.Path,required=True)
    ap.add_argument('--todo-provider',type=pathlib.Path,default=pathlib.Path('/home/tumlinson/.agents/skills/todo-orchestrator'))
    args=ap.parse_args()
    # Read the receipt first so missing prerequisites fail without touching authority.
    receipt=json.loads(args.receipt.read_text())
    sys.path.insert(0,str(args.todo_provider))
    from todo_orchestrator.semantic import SemanticReader
    state=SemanticReader(args.cellerator).state(task_id='CE-MOON-INTEGRATE')
    rows=state.get('tasks',[])
    done=any(t.get('id')=='CE-MOON-INTEGRATE' and t.get('effective_state')=='done' for t in rows)
    validate_receipt(receipt,args.cellerator,args.glasshelix,done)
    print('CE native capability receipt and source-bound evidence passed; authority read only')
if __name__=='__main__':
    try: main()
    except Exception as exc:
        print('CE receipt gate failed: '+str(exc),file=sys.stderr);sys.exit(1)
