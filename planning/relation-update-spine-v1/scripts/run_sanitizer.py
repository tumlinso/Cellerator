#!/usr/bin/env python3
"""Run required post-epic device sanitizer checks; missing tools/targets fail."""
from pathlib import Path
import argparse,json,re,shutil,subprocess,tempfile,sys
sys.dont_write_bytecode=True
PACKAGE=Path(__file__).resolve().parents[1]
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--binary-dir',type=Path,required=True)
    a.add_argument('--receipt',type=Path);o=a.parse_args()
    tool=shutil.which('compute-sanitizer')
    if not tool:raise ValueError('Compute Sanitizer unavailable; this is not a skipped pass')
    if o.receipt and o.receipt.exists():raise ValueError('receipt already exists')
    matrix=json.loads((PACKAGE/'machine/acceptance_matrix.json').read_text())
    cases=[t for t in matrix['tests'] if t['group'] in ['gpu','demo']]
    for t in cases:
        if not (o.binary_dir/t['binary']).is_file():raise ValueError('missing actual binary: '+t['binary'])
    runs=[]; completed=False
    try:
        with tempfile.TemporaryDirectory(prefix='ce-ru1-sanitizer-') as d:
            for t in cases:
                tools=['memcheck']
                if t['ctest_name'] in ['ru1_gpu_numerics','ru1_gpu_lifecycle','ru1_wmma_legality']:tools+=['racecheck','synccheck']
                for kind in tools:
                    log=Path(d)/(t['ctest_name']+'-'+kind+'.log')
                    cmd=[tool,'--tool',kind,'--error-exitcode','99','--log-file',str(log),str((o.binary_dir/t['binary']).resolve()),*t['argv']]
                    r=subprocess.run(cmd,text=True,capture_output=True,timeout=1800)
                    text=log.read_text() if log.exists() else ''
                    zero=bool(re.search(r'ERROR SUMMARY:\s*0 errors',text) if kind!='racecheck' else re.search(r'RACECHECK SUMMARY:\s*0 hazards.*\(0 errors, 0 warnings\)',text))
                    record={'test':t['ctest_name'],'tool':kind,'command':cmd,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'log':text,'passed':r.returncode==0 and zero}
                    runs.append(record);print(t['ctest_name'],kind,'PASS' if record['passed'] else 'FAIL')
                    if not record['passed']:raise ValueError('Sanitizer did not establish zero errors/hazards. Inspect raw log; never reinterpret as skip.')
        completed=True
    finally:
        if o.receipt:
            with o.receipt.open('x') as f:json.dump({'runs':runs,'passed':completed and len(runs)>0 and all(x['passed'] for x in runs)},f,indent=2)
if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.SubprocessError) as e:raise SystemExit('run_sanitizer: '+str(e))
