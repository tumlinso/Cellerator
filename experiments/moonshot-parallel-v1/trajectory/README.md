# Local trajectory references

This directory contains three CPU experiments from the moonshot vocabulary:

- M03: the exact local Sylvester flow `exp(dt L) X exp(dt R)`.
- M10: a linear delta ledger that remembers the last transmitted value.
- M02: the matrix polynomial `L X + X R + X M X`, with explicit JVP and VJP for state and factors.

The flow caches exponentials for inference under definition, structure and parameter generations, time step, dtype/device and actual parameter identity. Training computes fresh differentiable matrix exponentials on every call. The local flow does not solve a general nonlinear vector field or an externally coupled system.

The ledger stages each update before publishing its transmitted values and output. Thresholds compare against the last transmitted values, so gradual changes accumulate. Its componentwise instantaneous discrepancy estimate is `abs(W) @ abs(x - x_sent)`. Floating-point recurrence introduces additional rounding error; this estimate does not bound a trajectory. Zero-threshold updates reproduce the fixed linear map in real arithmetic, with the same floating-point recurrence caveat.

Checkpoints are JSON-compatible snapshots. Restoring a flow rebuilds derived exponentials; restoring a ledger preserves transmitted values and its accumulated output. The continuation tests compare fresh owners with uninterrupted execution.

Run the required CPU qualification with:

```sh
CUDA_VISIBLE_DEVICES='' /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B experiments/moonshot-parallel-v1/trajectory/check.py
```

These Torch/NumPy references establish small mathematical and continuation witnesses. Native registration, CUDA realization, scheduling, performance and scientific model selection require further work.
