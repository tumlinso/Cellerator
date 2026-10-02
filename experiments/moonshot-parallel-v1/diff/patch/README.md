# Native CPU patch derivatives

`patch.ops.forward(x,l,r,policy='fp32',generations=(0,0,0,0))`
returns `(y,tape)` for `Y=tanh(LX)R`. Matching nonempty square matrices
must use contiguous NumPy float32 storage. This experimental CPU API allows
square dimensions beyond the CUDA seed's fixed 16; it does not expose batching.

`vjp(tape,g,current_generations=None)` returns `(dx,dl,dr)`.
`jvp(tape,dx,dl,dr,current_generations=None)` returns the output direction.
Providing current generations rejects a mismatched structure/value/activity/
parameter generation. Tape operands and intermediates are immutable snapshots;
backward allocates outputs and does not update inputs or parameters.

`native.cc` supplies callable C++17 FP32 forward, VJP and JVP through ctypes.
The loader compiles a shared object under `/tmp/cellerator-moonshot-patch`, keyed
by source contents, compiler version and compilation flags. It requires a C++17
compiler (`CXX` or `c++`) and NumPy. No GPU backward or production registration
is provided here.

The `stored_half_ste` policy rounds X, L, R and V to binary16, retaining float32
storage for the CPU calculations. The first product T accumulates in FP32.
Its declared surrogate uses `1-tanh(T)^2` from saved T and stored V for the R
pullback. Input and parameter rounding are treated with a straight-through
surrogate. This is not the derivative of binary16 rounding, nor a claim of
bitwise CUDA equivalence. The `fp32` policy implements the smooth patch family.

Validation:

```
PYTHONPATH=experiments/moonshot-parallel-v1/diff \
  /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python \
  -B -m unittest patch.test_patch -v
```

Nine tests passed: every input/parameter gradient element via central finite
differences; JVP finite differences and the VJP/JVP adjoint identity; explicit
stored-half primal and surrogate oracles; trainable-zero support; immutable
snapshots, generation rejection and repeated nonmutating backward; malformed public tapes,
canonical immutable tape copies, native null/dimension checks and admission.
Finite differences qualify the smooth FP32 family. The stored-half policy is
checked against its declared surrogate and adjoint identity.

The exported C symbols are a private implementation ABI for the validated
Python wrapper. They reject null pointers, nonpositive/overflowing dimensions
and allocation failures. Raw pointers cannot prove allocation lengths or
aliasing; callers outside the wrapper must supply valid, disjoint output
storage. Public Tape construction validates every saved matrix and creates
immutable copies, and backward rechecks all saved extents before native calls.
