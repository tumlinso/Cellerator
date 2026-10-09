#!/usr/bin/env python3
"""Aggregate CPU algebra, compiled admission, or controller-owned GPU smoke."""
import argparse,hashlib,json,pathlib,subprocess,sys,datetime
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent

def cpu_checks():
    rng=np.random.default_rng(17)
    # Independent scalar stored-half oracle, including nonlinear half storage.
    for scale in (0.125,1.0,12.0):
        l,x,r=[(rng.normal(size=(16,16))*scale).astype(np.float16) for _ in range(3)]
        lf,xf,rf=[a.astype(np.float32) for a in (l,x,r)]
        t=lf@xf;v=np.tanh(t).astype(np.float16).astype(np.float32);y=v@rf
        scalar=np.empty((16,16),np.float32)
        for i in range(16):
            for j in range(16):
                total=np.float32(0)
                for k in range(16):
                    first=np.float32(0)
                    for q in range(16):first=np.float32(first+lf[i,q]*xf[q,k])
                    stored=np.float32(np.float16(np.tanh(first)))
                    total=np.float32(total+stored*rf[k,j])
                scalar[i,j]=total
        assert np.isfinite(y).all() and np.isfinite(scalar).all()
        np.testing.assert_allclose(y,scalar,rtol=2e-4,atol=2e-4)
    # Independent directional difference: zero and repeated ordered product args.
    for a,b,da,db,k in [(0.,3.,2.,-1.,4.),(2.,2.,3.,3.,.5),(2.,-3.,1.,4.,-2.)]:
        jvp=k*(da*b+a*db);eps=1e-5
        fd=(k*(a+eps*da)*(b+eps*db)-k*(a-eps*da)*(b-eps*db))/(2*eps)
        np.testing.assert_allclose(jvp,fd,rtol=1e-9,atol=1e-9)
    subprocess.run([sys.executable,'-B',str(HERE/'quad/check_mapping.py')],check=True)
    return 'CPU stored-half patch oracle, product zero/repeated directional witnesses and quad mapping passed; device kernels not executed'

def binaries(build):
    return [('patch16',build/'patch16/patch16_smoke',['--host-only'],[]),('quad',build/'quad/moonshot_quad_test',['--host'],[]),('product',build/'product/moon_product_smoke',[],['--gpu'])]

def source_hashes(module=None):
    modules=(module,) if module else ('patch16','quad','product')
    return {p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for name in modules for p in sorted((HERE/name).rglob('*')) if p.is_file() and (p.suffix in ('.cu','.cuh','.hpp','.cpp','.py') or p.name=='CMakeLists.txt')}

def build_input_hashes():
    result=source_hashes()
    for name in ('CMakeLists.txt','build_mma.py','check_mma.py','test_provenance.py','verify_gpu_evidence.py'):
        result[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return result

def load_build_manifest(build):
    path=build/'build-manifest.json'
    raw=path.read_bytes();manifest=json.loads(raw)
    assert manifest.get('schema')=='MMA-FRESH-BUILD/1','unbound build manifest'
    assert manifest.get('fresh_objects') is True,'build manifest does not attest fresh_objects=true'
    assert manifest.get('input_hashes')==build_input_hashes(),'build inputs changed since fresh compilation'
    assert manifest.get('source_hashes')==source_hashes(),'numerical sources changed since compilation'
    for name,binary,_,_ in binaries(build):
        assert manifest['binaries'][name]['sha256']==hashlib.sha256(binary.read_bytes()).hexdigest(),name+' binary differs from fresh build'
    return manifest,hashlib.sha256(raw).hexdigest()

def current_saved_manifest(results):
    """Return the portable manifest only while its complete source inputs remain current."""
    path=results/'build-manifest.json'
    try:
        raw=path.read_bytes();manifest=json.loads(raw)
    except (OSError,ValueError):
        return None,None
    if (manifest.get('schema')!='MMA-FRESH-BUILD/1' or manifest.get('fresh_objects') is not True
            or manifest.get('input_hashes')!=build_input_hashes()
            or manifest.get('source_hashes')!=source_hashes()):
        return None,None
    return manifest,hashlib.sha256(raw).hexdigest()

def _json_file(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None

def gpu_module_current(item,name,manifest,manifest_hash):
    return bool(manifest and isinstance(item,dict) and item.get('module')==name
        and item.get('stage')=='gpu' and item.get('passed') is True and item.get('exit_code')==0
        and item.get('source_hashes')==source_hashes(name)
        and item.get('build_manifest_sha256')==manifest_hash
        and item.get('binary_sha256')==manifest.get('binaries',{}).get(name,{}).get('sha256'))

def compiled_module_current(item,name,manifest,manifest_hash):
    return bool(manifest and isinstance(item,dict) and item.get('module')==name
        and item.get('stage')=='compiled_admission' and item.get('passed') is True and item.get('exit_code')==0
        and item.get('source_hashes')==source_hashes(name)
        and item.get('build_manifest_sha256')==manifest_hash
        and item.get('binary_sha256')==manifest.get('binaries',{}).get(name,{}).get('sha256'))

def gpu_validation_current(path,modules,results,manifest,manifest_hash):
    record=_json_file(path)
    if not (manifest and isinstance(record,dict) and record.get('stage')=='gpu'
            and record.get('passed') is True and record.get('source_hashes')==source_hashes()
            and record.get('build_manifest_sha256')==manifest_hash):
        return False
    recorded=record.get('modules',{})
    for name in modules:
        item=recorded.get(name)
        latest=_json_file(results/('gpu-'+name+'.json'))
        if not gpu_module_current(item,name,manifest,manifest_hash) or latest!=item:
            return False
    return True

def compiled_validation_current(path,modules,results,manifest,manifest_hash):
    record=_json_file(path)
    if not (manifest and isinstance(record,dict) and record.get('stage')=='compiled_admission'
            and record.get('passed') is True and record.get('source_hashes')==source_hashes()
            and record.get('build_manifest_sha256')==manifest_hash):
        return False
    recorded=record.get('modules',{})
    for name in modules:
        item=recorded.get(name)
        latest=_json_file(results/('compiled_admission-'+name+'.json'))
        if not compiled_module_current(item,name,manifest,manifest_hash) or latest!=item:
            return False
    return True

def reconcile_gpu_validation(validation,results,manifest,manifest_hash):
    """Drop passed GPU status whose source, build, binary, or run receipt is stale."""
    gpu={}
    for name in ('patch16','quad','product'):
        gpu[name]=gpu_module_current(_json_file(results/('gpu-'+name+'.json')),name,manifest,manifest_hash)
    for key,entry in list(validation.items()):
        if key=='gpu':modules=('patch16','quad','product')
        elif key.startswith('gpu-') and key[4:] in gpu:modules=(key[4:],)
        elif key=='compiled_admission':modules=('patch16','quad','product')
        elif key.startswith('compiled_admission-') and key[len('compiled_admission-'):] in gpu:modules=(key[len('compiled_admission-'):],)
        else:continue
        if not isinstance(entry,dict) or not entry.get('passed'):
            validation.pop(key,None)
            continue
        relative=entry.get('evidence','')
        if not isinstance(relative,str):
            validation.pop(key,None)
            continue
        relative_path=pathlib.Path(relative)
        if relative_path.parts and relative_path.parts[0]=='results':
            relative_path=pathlib.Path(*relative_path.parts[1:])
        path=(results/relative_path).resolve()
        if not path.is_relative_to(results.resolve()):
            validation.pop(key,None)
        elif key.startswith('compiled_admission'):
            if not compiled_validation_current(path,modules,results,manifest,manifest_hash):validation.pop(key,None)
        elif not gpu_validation_current(path,modules,results,manifest,manifest_hash):validation.pop(key,None)
    return gpu

def save_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-dir',type=pathlib.Path)
    ap.add_argument('--gpu',action='store_true',help='Only invoke through the root CUDA resource controller')
    ap.add_argument('--module',choices=('patch16','quad','product'),help='Run a single compiled module after its source changes')
    ap.add_argument('--record',action='store_true')
    args=ap.parse_args()
    require_build=bool(args.build_dir)
    assert not (args.gpu or args.module) or require_build,'--gpu/--module require --build-dir'
    stage='gpu' if args.gpu else ('compiled_admission' if require_build else 'cpu_reference')
    results=HERE/'results'
    manifest,manifest_hash=load_build_manifest(args.build_dir) if require_build else current_saved_manifest(results)
    if manifest_hash is not None:print('build_manifest_sha256='+manifest_hash,flush=True)
    output=[cpu_checks()]
    print(output[0],flush=True)
    record={'stage':stage,'passed':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output':output,'modules':{},'source_hashes':source_hashes(),'scope':'tiny correctness witnesses; no measured performance, native adapter or backward qualification'}
    if manifest is not None:
        record['build_manifest']=manifest
        record['build_manifest_sha256']=manifest_hash
    if args.record:results.mkdir(exist_ok=True)
    def preserve():
        if args.record:save_json(results/(stage+('-'+args.module+'-aggregate' if args.module else '')+'.json'),record)
    preserve()
    if require_build:
        for name,binary,hostflags,gpuflags in binaries(args.build_dir):
            if args.module and args.module!=name:continue
            assert binary.is_file(),f'{name} binary missing: {binary}'
            argv=[str(binary)]+(gpuflags if args.gpu else hostflags)
            result=subprocess.run(argv,capture_output=True,text=True)
            module={'module':name,'stage':stage,'passed':result.returncode==0,'argv':argv,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'source_hashes':source_hashes(name)}
            assert module['binary_sha256']==manifest['binaries'][name]['sha256'],name+' binary changed during execution'
            assert build_input_hashes()==manifest['input_hashes'],'build inputs changed during execution'
            module['build_manifest_sha256']=manifest_hash
            record['modules'][name]=module
            print(name+': '+result.stdout.strip(),flush=True)
            if result.stderr:print(result.stderr,file=sys.stderr,flush=True)
            if args.record:save_json(results/(stage+'-'+name+'.json'),module)
            preserve()
            # A skipped device test is not device validation; previous output is saved.
            assert result.returncode==0,f'{name} failed/skipped ({result.returncode}); prior modules preserved in results/{stage}-*.json'
            output.append(name+': '+result.stdout.strip())
    record['passed']=True
    preserve()
    if args.record:
        cap_path=HERE/'capability.json'
        cap=json.loads(cap_path.read_text()) if cap_path.exists() else {}
        previous=cap.get('source_hashes');cap['source_hashes']=record['source_hashes']
        if previous and previous!=record['source_hashes']:cap['validation']={}
        cap.setdefault('validation',{})[stage+('-'+args.module if args.module else '')]={'passed':True,'evidence':'results/'+stage+('-'+args.module+'-aggregate' if args.module else '')+'.json'}
        # Preserve valid per-module device evidence when another module changes.
        validation=cap.setdefault('validation',{})
        gpu=reconcile_gpu_validation(validation,results,manifest,manifest_hash)
        cap.update({'record_kind':'experimental_mma_capability','schema_version':1,'native_core_integration':False,'device_modules':gpu,'device_validation_scope':'per-module numerical smoke only; does not establish controller admission or full acceptance','full_acceptance_gate':'verify_gpu_evidence.py','forward':'GPU numerical smoke passed; full acceptance requires verify_gpu_evidence.py' if all(gpu.values()) else 'GPU numerical smoke partial or pending; see device_modules','derivatives':{'patch_input_vjp':'not_implemented','patch_parameter_vjp':'not_implemented','patch_jvp':'not_implemented','quad_derivatives':'not_implemented','product_jvp':'GPU smoke validated' if gpu['product'] else 'implemented source; GPU pending'},'precision':{'patch16':'FP16 L/X/R, FP32 accum, stored FP16 tanh intermediate, FP32 output','quad':'FP16 operands; FP32 accumulation/output','product':'FP32 product and directional derivative'},'timing':'not_measured','unsupported':['native prepared_program integration','CelleraTorch adapter','native backward routes','higher order differentiation','general shape/architecture support beyond module admission contracts']})
        save_json(cap_path,cap)
    print('stage='+stage+' passed',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:print('MMA aggregate failed: '+str(exc),file=sys.stderr);sys.exit(1)
