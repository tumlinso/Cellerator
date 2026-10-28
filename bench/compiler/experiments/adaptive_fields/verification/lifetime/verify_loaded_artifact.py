#!/usr/bin/env python3
"""Independently call the CPU JIT probe's emitted shared object vs FP64 math."""

import ctypes
import math
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_loaded_artifact.py ARTIFACT.so")
    lib = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
    run = lib.cellerator_region
    run.argtypes = [
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
    ]
    run.restype = ctypes.c_int

    off = (ctypes.c_uint64 * 5)(0, 2, 3, 5, 6)
    col = (ctypes.c_uint64 * 6)(0, 2, 1, 0, 2, 1)
    seg = (ctypes.c_uint64 * 3)(0, 2, 4)
    values = (ctypes.c_float * 6)(0.5, 0.25, 1.0, 0.75, 0.5, 2.0)
    x = (ctypes.c_float * 3)(2.0, 3.0, 4.0)
    out = (ctypes.c_float * 4)()
    status = run(off, col, values, x, out, 4, seg, 2)
    if status != 0:
        raise AssertionError(f"module returned {status}")

    ax = []
    for row in range(4):
        ax.append(sum(float(values[e]) * float(x[col[e]])
                      for e in range(off[row], off[row + 1])))
    expected = [0.0] * 4
    for begin, end in ((0, 2), (2, 4)):
        scores = ax[begin:end]
        top = max(scores)
        exp_scores = [math.exp(score - top) for score in scores]
        denom = sum(exp_scores)
        expected[begin:end] = [v / denom for v in exp_scores]

    errors = [abs(float(out[i]) - expected[i]) for i in range(4)]
    if max(errors) > 2e-6:
        raise AssertionError(f"FP64 mismatch: output={list(out)}, expected={expected}, errors={errors}")
    print(f"artifact={Path(sys.argv[1]).resolve()} output={list(out)} "
          f"fp64_max_abs_error={max(errors):.9g} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
