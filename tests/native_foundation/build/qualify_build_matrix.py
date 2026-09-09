#!/usr/bin/env python3
"""Build and fingerprint the actual NF1 host/CUDA owners; never execute a GPU."""
import argparse, hashlib, json, pathlib, subprocess

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=pathlib.Path,required=True)
    p.add_argument('--output-dir',type=pathlib.Path,required=True)
    p.add_argument('--cuda-compiler',required=True)
    p.add_argument('--cuda-host-compiler',required=True)
    p.add_argument('--baseplane-source',required=True)
    p.add_argument('--architecture',default='70')
    p.add_argument('--jobs',type=int,default=8)
    a=p.parse_args(); root=a.repo_root.resolve(); out=a.output_dir.resolve()
    if out.is_relative_to(root): raise ValueError('matrix evidence must be outside source')
    out.mkdir(parents=True,exist_ok=True); commands=[]
    def run(argv):
        r=subprocess.run([str(v) for v in argv],text=True,capture_output=True)
        commands.append({'argv':[str(v) for v in argv],'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if r.returncode: raise RuntimeError(r.stdout+r.stderr)
        return r.stdout
    head=run(['git','-C',root,'rev-parse','HEAD']).strip()
    if run(['git','-C',root,'status','--porcelain=v1','--untracked-files=all']): raise ValueError('clean committed source required')
    host=out/'host'; cuda=out/'cuda'
    run(['cmake','-S',root,'-B',host,'-DCELLERATOR_ENABLE_CUDA=OFF','-DCELLERATOR_NATIVE_FOUNDATION_ONLY=ON','-DCELLERATOR_BUILD_NATIVE_FOUNDATION_TESTS=ON','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_CUDA_COMPILER=/unavailable/nvcc'])
    run(['cmake','--build',host,'--target','cellerator_nf1_host_correctness','-j',a.jobs])
    run(['cmake','-S',root,'-B',cuda,'-DCELLERATOR_ENABLE_CUDA=ON','-DCELLERATOR_BUILD_SEMANTIC_SPINE_V1=ON','-DCELLERATOR_BUILD_TESTS=OFF','-DCELLERATOR_BUILD_SEMANTIC_SPINE_EXAMPLES=OFF','-DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_CUDA_COMPILER='+a.cuda_compiler,'-DCMAKE_CUDA_HOST_COMPILER='+a.cuda_host_compiler,'-DCMAKE_CUDA_ARCHITECTURES='+a.architecture,'-DCUDAToolkit_ROOT='+str(pathlib.Path(a.cuda_compiler).parent.parent),'-DBASEPLANE_SOURCE_DIR='+a.baseplane_source])
    targets=['cellerator_prepared_relation_cuda','cellerator_relation_algebra','cellerator_runtime']
    run(['cmake','--build',cuda,'--target',*targets,'-j',a.jobs])
    ldd=run(['ldd',host/'nf1-build-tests/ce_nf1_b01'])
    if any(x in ldd.lower() for x in ['libcuda','libcudart','libtorch','cellshard','baseplane']): raise ValueError('forbidden host dependency')
    host_commands=json.loads((host/'compile_commands.json').read_text())
    if any(x['file'].endswith('.cu') or 'nvcc' in x['command'] for x in host_commands): raise ValueError('CUDA leaked into host compilation')
    def record(path): return {'path':str(path),'sha256':digest(path)}
    artifacts=[record(cuda/('lib'+t+'.a')) for t in targets]
    artifacts += [record(host/('lib'+t+'.a')) for t in ['cellerator_operation_schema_v2','cellerator_prepared_program_v2','cellerator_relation_semantics','cellerator_relation_calculus','cellerator_segment_host','cellerator_gate_validation']]
    artifacts += [record(d/f) for d in [host,cuda] for f in ['CMakeCache.txt','compile_commands.json']]
    manifest=host/'CelleratorNativeFoundationDependencies.json'
    if run(['git','-C',root,'rev-parse','HEAD']).strip()!=head or run(['git','-C',root,'status','--porcelain=v1','--untracked-files=all']): raise ValueError('source changed during matrix')
    receipt={'kind':'nf1-build-matrix-v1','source_commit':head,'repo_root':str(root),'host_build':str(host),'cuda_build':str(cuda),'architecture':a.architecture,'cuda_compiler':a.cuda_compiler,'cuda_version_output':run([a.cuda_compiler,'--version']),'artifacts':artifacts,'dependency_manifest':record(manifest),'commands':commands,'qualification':'compiled CUDA owners and executed host tests only; no device execution or performance claim','passed':True}
    (out/'matrix.json').write_text(json.dumps(receipt,indent=2)+'\n');print(out/'matrix.json')
if __name__=='__main__': main()
