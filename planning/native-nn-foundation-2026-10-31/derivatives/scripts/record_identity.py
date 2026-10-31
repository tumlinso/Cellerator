#!/usr/bin/env python3
"""Record source/binary identities for the bounded derivative qualification."""
import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parents[4]
evidence = root / 'planning/native-nn-foundation-2026-10-31/derivatives/evidence'
paths = [
    'include/Cellerator/nn/ops.hh', 'src/nn/ops_cpu.cc', 'src/nn/ops_cuda.cc',
    'tests/nn/native/ops_cpu.cc', 'tests/nn/native/ops_cuda.cc',
    'include/Cellerator/modeling/cuda/attention.hh',
    'include/Cellerator/modeling/cuda/grouped_linear.hh',
    'include/Cellerator/modeling/cuda/runtime.hh',
    'src/modeling/cuda/attention.cu', 'src/modeling/cuda/grouped_linear.cu',
    'src/modeling/cuda/runtime.cu', 'src/compute/operation/product2/product2.cc',
    'planning/native-nn-foundation-2026-10-31/derivatives/scripts/CMakeLists.txt',
]
identity = {'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
            'files': {}, 'binaries': {}, 'qualification': 'correctness and lifetime only; no timings'}
for path in paths:
    identity['files'][path] = hashlib.sha256((root/path).read_bytes()).hexdigest()
for name in ['native_nn_ops_cpu_test', 'native_nn_ops_cuda_test']:
    p = pathlib.Path('/tmp/cellerator-native-derivatives')/name
    if p.exists(): identity['binaries'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
(evidence / (sys.argv[1] if len(sys.argv)>1 else 'identity.json')).write_text(json.dumps(identity,indent=2)+'\n')
