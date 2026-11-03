from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--sanitizer',choices=['memcheck','synccheck','initcheck','racecheck']);a=p.parse_args()
suppression=Path('/tmp/cellerator-native-data-implementation/frozen/2f06982e2b948b0d7412a723a8d20a34ee1e0c23a89c5b12539eb1299c715378/peer_access_already_enabled.xml')
if hashlib.sha256(suppression.read_bytes()).hexdigest()!='2f06982e2b948b0d7412a723a8d20a34ee1e0c23a89c5b12539eb1299c715378':raise RuntimeError('sanitizer suppression changed')
cases=[('distributed', '9b878ccbf31bb93e12637785c435df5f32756d48dcde5ff6a1a9ab459abce2fe', 'cellerator_native_data_distributed_cuda_test', ['--device0', '1', '--device1', '0']), ('32way', 'e699fab56e24ab9b1c5920886873a835e8cf6f87d4dbeb080aa39ebf7692fd83', 'cellerator_native_data_distributed_32way_test', ['--device0', '1', '--device1', '0']), ('program_contract', '154fcc3abb7fe65831ed5dad1ed34ba16965b81bc4ce846662f43ec1922c016b', 'cellerator_native_data_packed_program_contract_test', []), ('packed_local', 'a7374588957039e1cf1f8783655e65824a8c5736fe2de60d5ca88ebafcb55c83', 'cellerator_native_data_block_masked_compile_test', ['--device', '0']), ('transfer', '3021cbd3e76781faf54664fc8285c22bbfc6a4e28c99ffa1e4bfe905e5d17775', 'cellerator_native_data_transfer_compile_test', []), ('native_owner', '8981071ed1b70e42849116e3ae954bcd6e16939600154d5514315399e0aa1c3c', 'cellerator_native_owner_cuda_test', ['--device', '1'])]
expected='GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a,GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505'
if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected:raise RuntimeError('unexpected Scheduler-visible device order')
for label,digest,binary,args in cases:
 path=Path('/tmp/cellerator-native-data-implementation/frozen')/digest/binary
 if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('binary identity changed: '+label)
 command=[str(path),*args]
 if a.sanitizer:command=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer','--tool',a.sanitizer,'--error-exitcode','97','--print-limit','8','--suppressions',str(suppression),*(['--leak-check','full'] if a.sanitizer=='memcheck' else []),*command]
 print(json.dumps({'case':label,'sha256':digest,'sanitizer':a.sanitizer,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}),flush=True)
 begin=time.monotonic();r=subprocess.run(command,check=False)
 print(json.dumps({'case':label,'exit_code':r.returncode,'elapsed_seconds':time.monotonic()-begin}),flush=True)
 if r.returncode:sys.exit(r.returncode)
print('all frozen native-data qualification cases passed',flush=True)
