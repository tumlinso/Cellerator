#!/usr/bin/env python3
"""Run inside an admitted Scheduler job; retain each CUDA sanitizer log."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tool', required=True, help='Absolute compute-sanitizer path')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('argv', nargs=argparse.REMAINDER)
    args = p.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if not argv:
        p.error('supply the checked executable and arguments after --')
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for tool in ('memcheck', 'racecheck', 'synccheck'):
        command = [args.tool, '--tool', tool, '--error-exitcode', '91', *argv]
        started = time.monotonic()
        run = subprocess.run(command, capture_output=True, text=True, timeout=90)
        (args.output/f'{tool}.stdout').write_text(run.stdout)
        (args.output/f'{tool}.stderr').write_text(run.stderr)
        result = dict(tool=tool, command=command, exit_code=run.returncode,
                      wall_seconds=time.monotonic()-started)
        results.append(result)
        print(json.dumps(result), flush=True)
        if run.returncode:
            print(run.stdout, run.stderr, flush=True)
            break
    summary = dict(results=results, passed=len(results) == 3 and
        all(x['exit_code'] == 0 for x in results),
        binary_sha256=hashlib.sha256(Path(argv[0]).read_bytes()).hexdigest(),
        sanitizer_sha256=hashlib.sha256(Path(args.tool).read_bytes()).hexdigest())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
