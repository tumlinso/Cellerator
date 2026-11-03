"""Protocol-only stand-in for CPU smoke testing the validation wrapper.

This is the oracle itself and must never be treated as a candidate or timing
backend. Real candidate qualification uses the native CUDA runner.
"""

import argparse
from pathlib import Path

from case_io import read_case, write_result
from reference import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("packet", "direct"), required=True)
    parser.add_argument("--gradient-mode", choices=("overwrite", "accumulate"),
                        default="overwrite")
    parser.add_argument("--repeat-vjp", action="store_true")
    args = parser.parse_args()
    problem, dy, output_dtype = read_case(args.input)
    expected = evaluate(problem, dy, output_dtype)
    if args.gradient_mode == "accumulate":
        for name in ("dx", "dz", "dalpha", "dbias", "dgamma"):
            getattr(expected.vjp, name)[:] += 0.125
    write_result(args.output, problem, expected.y, expected.vjp, output_dtype,
                 expected.vjp if args.repeat_vjp else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
