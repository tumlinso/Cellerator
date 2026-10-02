#!/usr/bin/env python3
"""Pure gate: match numerical sources to recorded device evidence; no GPU launch."""
import hashlib,json,pathlib,re,sys
import check_mma
HERE=pathlib.Path(__file__).resolve().parent

def require(ok,message):
    if not ok:raise ValueError(message)

def verify():
    results=HERE/'results';aggregate=json.loads((results/'gpu.json').read_text());receipt=json.loads((results/'controller-success.json').read_text())
    require(aggregate.get('stage')=='gpu' and aggregate.get('passed') is True,'latest aggregate GPU run failed or absent')
    require(receipt.get('ok') is True and receipt.get('returncode')==0 and receipt.get('status')=='succeeded','controller did not accept this execution')
    require(bool(receipt.get('evidence_id')),'missing controller evidence identity')
    require(aggregate.get('source_hashes')==check_mma.source_hashes(),'numerical sources changed since aggregate GPU check')
    stdout=''
    for kind in ('stdout','stderr'):
        artifact=receipt['preserved_artifacts'][kind];path=results/artifact['path']
        require(path.parent==results and not path.is_symlink(),'unsafe preserved controller path')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==artifact['sha256'],'controller '+kind+' hash changed')
        if kind=='stdout':stdout=path.read_text()
    require('stage=gpu passed' in stdout,'controller output lacks completed aggregate check')
    lease=json.loads((results/receipt['lease_evidence']).read_text())
    require(lease.get('format')=='CUDA-FOREGROUND-LEASE/1' and bool(lease.get('resource_ids')),'assigned CUDA lease evidence missing')
    for module in ('patch16','quad','product'):
        item=json.loads((results/('gpu-'+module+'.json')).read_text())
        require(item==aggregate.get('modules',{}).get(module),'latest '+module+' evidence differs from aggregate execution')
        require(item.get('module')==module and item.get('stage')=='gpu' and item.get('passed') is True and item.get('exit_code')==0,module+' device check incomplete')
        require(item.get('source_hashes')==check_mma.source_hashes(module),module+' source changed')
        require(bool(re.fullmatch('[0-9a-f]{64}',item.get('binary_sha256',''))),module+' binary identity missing')
        require(item.get('stdout','').strip() in stdout,module+' output is not bound to controller transcript')
        text=item['stdout']
        if module=='patch16':
            for count in (1,4,5):
                for kind in (0,1,2):require(f'patch16 oracle passed count={count} kind={kind}' in text,'missing stored-half patch fixture')
        elif module=='quad':require('quad GPU PASS' in text and 'sm=70' in text and 'fixtures=198 outputs=101376' in text,'quad mapping coverage missing')
        else:
            for count in (0,1,31,32,33,127,129):require(f'product GPU count={count} PASS' in text,'product extent/response fixture missing')
    # Independent host algebra/mapping checks. This function performs no GPU calls.
    print(check_mma.cpu_checks())
    print('GPU evidence gate passed: all three current numerical sources match controller evidence '+receipt['evidence_id']+'; no GPU relaunched')
if __name__=='__main__':
    try:verify()
    except Exception as exc:print('GPU evidence gate failed: '+str(exc),file=sys.stderr);sys.exit(1)
