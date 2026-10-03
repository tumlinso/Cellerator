#!/usr/bin/env python3
"""Install actual components and validate consumers against a relocated SDK."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
p = argparse.ArgumentParser()
p.add_argument('--build-dir', required=True)
p.add_argument('--prefix', required=True)
p.add_argument('--require-integrated', action='store_true')
p.add_argument('--enable-sm70', action='store_true')
p.add_argument('--cuda-compiler')
p.add_argument('--cuda-host-compiler')
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
build = Path(a.build_dir).resolve()
prefix = Path(a.prefix).resolve()
def run(argv):
    result = subprocess.run([str(x) for x in argv], text=True, capture_output=True)
    print("$ " + " ".join(str(x) for x in argv), flush=True)
    lines = (result.stdout + result.stderr).splitlines()
    print("\n".join(lines[-50:] if result.returncode else lines[-10:]), flush=True)
    result.check_returncode()
configure = ['cmake', '-S', root/'cmake/substrate', '-B', build,
             '-DCMAKE_INSTALL_PREFIX='+str(prefix),
             '-DCELLERATOR_SUBSTRATE_REQUIRE_INTEGRATED='+('ON' if a.require_integrated else 'OFF'),
             '-DCELLERATOR_SUBSTRATE_ENABLE_SM70='+('ON' if a.enable_sm70 else 'OFF')]
if a.cuda_compiler:
    configure.append('-DCMAKE_CUDA_COMPILER='+a.cuda_compiler)
if a.cuda_host_compiler:
    configure.append('-DCMAKE_CUDA_HOST_COMPILER='+a.cuda_host_compiler)
run(configure)
run(['cmake', '--build', build, '--parallel', '2'])
run(['cmake', '--install', build])
with tempfile.TemporaryDirectory(prefix='ce-substrate-relocated-') as temporary:
    relocated = Path(temporary)/'sdk'
    shutil.copytree(prefix, relocated)
    # Installed CMake/headers must not reference producer-private source trees.
    for path in list(relocated.rglob('*.cmake')) + list(relocated.rglob('*.hh')):
        if str(root) in path.read_text(errors='replace'):
            raise RuntimeError('producer source path leaked into installed SDK: '+str(path))
    unavailable = build/'unavailable-component'
    unavailable.mkdir(parents=True, exist_ok=True)
    (unavailable/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.24)\n'
        'project(MissingFullOwner LANGUAGES CXX)\n'
        'find_package(Cellerator CONFIG REQUIRED COMPONENTS indexed_mechanism)\n')
    missing = subprocess.run(['cmake', '-S', str(unavailable), '-B', str(unavailable/'build'),
                              '-DCMAKE_PREFIX_PATH='+str(relocated)], text=True, capture_output=True)
    if missing.returncode == 0 or 'FOUND to FALSE' not in missing.stderr:
        raise RuntimeError('unavailable full indexed owner did not fail package discovery')
    print('Unavailable full indexed_mechanism correctly rejected', flush=True)
    for profile in ['baseline', 'effects-only']:
        consumer = build/('consumer-'+profile)
        run(['cmake', '-S', root/'examples/substrate/install', '-B', consumer,
             '-DCMAKE_PREFIX_PATH='+str(relocated),
             '-DCE_REQUIRE_INTEGRATED='+('ON' if a.require_integrated and profile=='baseline' else 'OFF'),
             '-DCE_EFFECTS_ONLY='+('ON' if profile=='effects-only' else 'OFF')])
        run(['cmake', '--build', consumer, '--parallel', '2'])
        run(['ctest', '--test-dir', consumer, '--output-on-failure'])

if not a.require_integrated and not (root/'include/Cellerator/state/structured_state.hh').exists():
    guard = subprocess.run(['cmake', '-S', str(root/'cmake/substrate'), '-B', str(build/'integration-guard'),
                            '-DCELLERATOR_SUBSTRATE_REQUIRE_INTEGRATED=ON'], text=True, capture_output=True)
    if guard.returncode == 0 or 'integration prerequisite missing' not in guard.stderr:
        raise RuntimeError('missing integrated source guard did not reject configuration')
    print('Missing integrated source correctly rejected at configuration', flush=True)
