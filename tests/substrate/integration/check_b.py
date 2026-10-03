#!/usr/bin/env python3
"""Selected native host components and one installed-prefix composition consumer."""
import argparse
from pathlib import Path
import subprocess
p = argparse.ArgumentParser()
p.add_argument('--build-dir', required=True)
p.add_argument('--sdk-prefix', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
build = Path(a.build_dir).resolve()
prefix = Path(a.sdk_prefix).resolve()
commands = [
    ['cmake', '-S', str(root / 'cmake/substrate'), '-B', str(build / 'owners'),
     '-DCELLERATOR_SUBSTRATE_COMPONENTS=packing_strategies;frontier;response;adaptive',
     '-DCELLERATOR_SUBSTRATE_ENABLE_SM70=OFF', '-DCMAKE_BUILD_TYPE=Release',
     '-DCMAKE_INSTALL_PREFIX=' + str(prefix)],
    ['cmake', '--build', str(build / 'owners'), '--parallel', '2'],
    ['cmake', '--install', str(build / 'owners')],
    ['cmake', '-S', str(root / 'tests/substrate/integration'), '-B', str(build / 'consumer'),
     '-DCMAKE_PREFIX_PATH=' + str(prefix)],
    ['cmake', '--build', str(build / 'consumer'), '--parallel', '2'],
    ['ctest', '--test-dir', str(build / 'consumer'), '--output-on-failure', '-V'],
]
for command in commands:
    subprocess.run(command, check=True)
