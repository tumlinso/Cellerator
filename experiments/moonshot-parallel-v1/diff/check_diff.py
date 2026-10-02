"""Run CPU derivative checks and verify immutable source-bound capability."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RECEIPT = ROOT / 'capability-receipt.json'
SUITES = ['patch.test_patch', 'product.test_product', 'ports.test_ports', 'adapter.test_adapter']


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*'))
            if p.is_file() and (p.suffix in {'.py', '.cc', '.cpp', '.h', '.hh'} or p.name == 'README.md')}


def declarations():
    return {
        'route': 'C++17 CPU native numerical execution via ctypes; experimental Torch adapter',
        'forward': True, 'input_vjp': True, 'parameter_vjp': True,
        'selected_explicit_jvp': True, 'backward_mutates_parameters': False,
        'saved_primal': 'native owned immutable copies; Torch saved original input version checks',
        'precision': {'patch': ['fp32', 'stored_half_ste'], 'product': ['fp32'], 'ports': ['fp32']},
        'mixed_derivative_convention': 'stored half operands/intermediate; straight-through quantization surrogate',
        'shapes': {'patch': 'matching nonempty square matrices',
                   'product': 'flattened vectors; ordered and repeated indexed pairs',
                   'ports': 'flattened heterogeneous positive actor widths; shared port width; supplied directed edges'},
        'torch_admission': 'CPU float32 contiguous tensors and static integer metadata',
        'production_celleratorch_registration': False,
        'unsupported': ['CUDA derivatives', 'Torch forward AD', 'higher-order derivatives', 'batched operators'],
        'cuda_forward_evidence': 'separate MMA lane; no CUDA derivative qualification',
    }


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--record',action='store_true'); args=parser.parse_args()
    sources=source_hashes()
    if not args.record:
        receipt=json.loads(RECEIPT.read_text())
        if receipt.get('source_sha256') != sources:
            raise SystemExit('Derivative capability receipt source hashes do not match current sources')
        if receipt.get('capability') != declarations() or receipt.get('suites') != SUITES:
            raise SystemExit('Derivative capability receipt declarations or suites mismatch')
        if receipt.get('qualification') != 'passed':
            raise SystemExit('Derivative capability receipt lacks passing qualification')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for suite in SUITES:
        subprocess.run([sys.executable,'-B','-m','unittest','-v',suite],cwd=ROOT,env=env,check=True)
    if source_hashes()!=sources:
        raise SystemExit('Sources changed during derivative qualification')
    if args.record:
        import numpy
        import torch
        compiler=os.environ.get('CXX','c++')
        compiler_identity=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
        receipt={'schema_version':1,'qualification':'passed','suites':SUITES,
                 'source_sha256':sources,'capability':declarations(),
                 'environment':{'python':platform.python_version(),'numpy':numpy.__version__,
                                'torch':torch.__version__,'platform':platform.platform(),'device':'CPU',
                                'native_compiler':compiler_identity,
                                'native_compile_flags':{'patch':['-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC'],
                                                        'product':['-std=c++17','-O2','-shared','-fPIC','-ffp-contract=off'],
                                                        'ports':['-std=c++17','-O2','-shared','-fPIC','-ffp-contract=off']},
                                'native_cache_identity':'source contents, compiler identity and numerical compile flags; wrappers source-bound above'}}
        RECEIPT.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('CPU native derivative and experimental Torch adapter qualification passed')

if __name__=='__main__': main()
