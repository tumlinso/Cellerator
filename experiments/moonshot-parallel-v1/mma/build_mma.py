#!/usr/bin/env python3
"""Build from fresh objects and bind exact build inputs to executable identities."""
import argparse, datetime, hashlib, json, pathlib, subprocess
import check_mma
HERE=pathlib.Path(__file__).resolve().parent

def build(output,compiler,jobs=2):
    output=output.resolve()
    # Never reuse objects or an existing manifest to certify current source.
    output.mkdir(parents=True,exist_ok=False)
    before=check_mma.build_input_hashes()
    commands=[['cmake','-S',str(HERE),'-B',str(output),'-DMOONSHOT_CUDA=ON',
               '-DCMAKE_CUDA_COMPILER='+str(compiler),'-DCMAKE_BUILD_TYPE=Release'],
              ['cmake','--build',str(output),'-j'+str(jobs)]]
    version=subprocess.run([str(compiler),'--version'],capture_output=True,text=True,check=True).stdout
    logs=[]
    for i,argv in enumerate(commands):
        result=subprocess.run(argv,capture_output=True,text=True)
        text=result.stdout+result.stderr
        log=output/('configure.log' if i==0 else 'build.log');log.write_text(text)
        print(text,flush=True)
        result.check_returncode()
        logs.append({'argv':argv,'exit_code':result.returncode,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
    assert before==check_mma.build_input_hashes(),'build inputs changed while compiling'
    manifest={'schema':'MMA-FRESH-BUILD/1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'fresh_objects':True,'input_hashes':before,'source_hashes':check_mma.source_hashes(),
              'compiler':str(compiler),'compiler_version':version,'architecture':'sm_70','commands':logs,
              'binaries':{name:{'path':binary.relative_to(output).as_posix(),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()} for name,binary,_,_ in check_mma.binaries(output)}}
    check_mma.save_json(output/'build-manifest.json',manifest)
    # Preserve portable evidence outside ignored executable/object directories.
    check_mma.save_json(HERE/'results/build-manifest.json',manifest)
    print('Fresh build bound to source and binary hashes: '+str(output),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=pathlib.Path,required=True)
    parser.add_argument('--compiler',type=pathlib.Path,default=pathlib.Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc'))
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args();build(args.build_dir,args.compiler,args.jobs)
