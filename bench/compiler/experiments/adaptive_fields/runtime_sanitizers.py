#!/usr/bin/env python3
"""Run the frozen E6/E7 runtime sanitizer probes inside one admitted job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / 'gpu/runtime'
TOOL = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer')


def checked(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Artifact changed before sanitizer invocation: {path}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', choices=('all', 'compiler', 'lifetime'), default='all')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'evidence/runtime_sanitizers')
    args = parser.parse_args()
    # These are accidental-direct-run guards, not resource admission evidence.
    # The installed scheduler supplies UUID routing but no job/attempt variables;
    # actual admission is established by the externally retained scheduler receipt.
    if (os.environ.get('CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED') != '1'
            or not os.environ.get('ADAPTIVE_FIELDS_RUN_KEY')
            or not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')):
        raise RuntimeError('Use the root-owned scheduler job specification')
    compiler = json.loads((RUNTIME / 'artifacts/nvcc_artifact_manifest.json').read_text())
    lifetime = json.loads((RUNTIME / 'lifetime/artifacts/build_manifest.json').read_text())
    for path, expected in [
        (Path(compiler['runner_executable']), compiler['runner_sha256']),
        (Path(compiler['artifact']), compiler['artifact_sha256']),
        (RUNTIME / 'region.cu', compiler['source_sha256']),
        (Path(lifetime['runner']), lifetime['runner_sha256']),
        (Path(lifetime['kernel_source']), lifetime['kernel_source_sha256']),
    ]:
        checked(path, expected)
    output = args.output_dir.resolve()
    if HERE not in output.parents:
        raise RuntimeError('Sanitizer evidence must stay inside the experiment')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    selected = ('compiler', 'lifetime') if args.probe == 'all' else (args.probe,)
    for label in selected:
        for tool in ('memcheck', 'racecheck', 'synccheck'):
            result_path = output / f'{label}_{tool}.json'
            if label == 'compiler':
                argv = [compiler['runner_executable'], str(RUNTIME / 'region.cu'),
                        compiler['artifact'], str(result_path), 'compute_70']
            else:
                argv = [lifetime['runner'], lifetime['kernel_source'], str(result_path)]
            argv = [str(TOOL), '--tool', tool, '--error-exitcode', '99', *argv]
            run = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=480)
            record = {'probe': label, 'tool': tool, 'argv': argv,
                      'returncode': run.returncode, 'stdout': run.stdout,
                      'stderr': run.stderr, 'result_path': str(result_path)}
            records.append(record)
            (output / f'{label}_{tool}_log.json').write_text(json.dumps(record, indent=2) + '\n')
            print(f'{label}/{tool}: exit {run.returncode}', flush=True)
            if run.returncode:
                break
        if records[-1]['returncode']:
            break
    passed = len(records) == 3 * len(selected) and all(r['returncode'] == 0 for r in records)
    (output / 'summary.json').write_text(json.dumps({
        'passed': passed, 'checks': records,
        'timings_are_performance_evidence': False,
        'scheduler_correlation_key': os.environ['ADAPTIVE_FIELDS_RUN_KEY'],
        'admission_evidence': 'External JobSpec, submission and terminal receipt',
    }, indent=2) + '\n')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
