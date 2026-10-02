# Native CPU product derivative packet

`product.ops.forward(x, k, a, b, generations=(0,0,0,0))` returns `(y, tape)`
for ordered `y[i]=(k[i]*x[a[i]])*x[b[i]]`. Inputs and coefficients are rank-one
float32 arrays; both index arrays are rank-one int64 arrays with one entry per
process. Process outputs remain distinct. Shared coefficient IDs are gathered
by the caller and their cotangents reduced there.

`vjp(tape,g,current_generations=None)` returns logical-input scatter-add `dx`
and one `dk` per process. `jvp(tape,dx,dk,current_generations=None)` returns the
combined input/parameter direction. Repeated inputs retain multiplicity, including
`a==b`; zero arguments use direct products without division. Empty packets work.

The tape owns immutable bytes-backed copies of the actual float32 operands and
indices, with `(structure,value,activity,parameter)` generation labels. Supplying
current labels rejects stale reuse. Without current labels, derivatives evaluate
the saved snapshot. The caller owns external generation tracking and synchronization.
Backward does not change a primal, parameter, tape or owner.

The native C++17 ABI validates nonnegative extents, required pointers and every
logical index before writing outputs. Python checks dtype, rank and matching
extents and allocates independent outputs. C++ raw-pointer callers must supply
the declared storage lengths and nonaliasing output buffers. The wrapper compiles
a captured source snapshot into a shared library under `/tmp`, keyed by source,
compiler identity and flags, using a file lock and atomic installation.

Forward uses float32 multiply order from the accepted MMA oracle. JVP uses the
oracle's explicit `fma` state term plus the coefficient direction. VJP and JVP
use the ordinary differentiable algebra evaluated on saved float32 operands;
they do not differentiate discontinuous floating-point rounding. No production
registration, CUDA derivative execution or timing claim is provided here.

Run from the repository root:

```sh
PYTHONPATH=experiments/moonshot-parallel-v1/diff /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B experiments/moonshot-parallel-v1/diff/product/test_product.py
```

Five tests cover independent finite differences for every input and parameter,
combined-direction finite differences, VJP/JVP adjoint identity, repeated/zero
paths, empty packets, immutable saved-primal reuse, generation rejection and
Python/native admission before output mutation.
