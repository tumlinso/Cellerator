# `compute/ml`

ML-facing sparse math built from `compute/sparse` and low-level CUDA kernels.

This folder owns sparse training primitives, autograd surfaces, model-adjacent
operators, and preprocessing math kernels. It does not own biology policy,
storage policy, or neighbor-calling workflow policy.
