# Python bindings

The Python distribution is named `cellerator`. It provides Torch-free identity
and mechanism declarations plus native numerical entry points. Torch support is
an optional adapter in `cellerator.torch`; it does not create another topology
or parameter owner. No compiler API is exposed through this binding surface.

## Build and install

Native C++ consumers keep both binding options off by default. The host Python
wheel enables Python and Product2 while leaving CUDA and Torch disabled. The root
`pyproject.toml` configures a Python build with Torch disabled, so install the
core distribution with:

```sh
python -m pip install .
```

For the optional Torch adapter, install the distribution's Torch extra and
configure both Torch and the CUDA backend. This is the only mode that discovers
and links libtorch; the Torch adapter requires the CUDA indexed-mechanism owner:

```sh
python -m pip install '.[torch]' \
  --config-settings=cmake.define.CELLERATOR_ENABLE_CUDA=ON \
  --config-settings=cmake.define.CELLERATOR_ENABLE_TORCH=ON
```

For direct CMake builds, set `CELLERATOR_ENABLE_PYTHON=ON` to build the
`_native` extension and configure `CELLERATOR_ENABLE_CUDA=ON` plus
`CELLERATOR_ENABLE_TORCH=ON` to add the optional Torch adapters. Both binding
options remain off for ordinary C++ builds.

The selected native CPU or CUDA backend determines which mechanism capabilities
are available. Python reports mechanism bindings as unavailable when their
native targets are not part of the build. Core product2 remains CPU FP32.

Installed C++ consumers that need the adapter request it explicitly:

```cmake
find_package(Cellerator CONFIG REQUIRED COMPONENTS torch)
target_link_libraries(my_target PRIVATE Cellerator::torch)
```

Ordinary `find_package(Cellerator)` consumers do not discover or link Torch.

## API shape

Import identity and mechanism declarations from `cellerator`. Product2 accepts
NumPy vectors through `cellerator._native.product2_forward`,
`product2_vjp`, and `product2_jvp`; the operation retains its native FP32
contract. Mechanism preparation takes a typed mechanism specification and contiguous
NumPy FP32 initial coefficients, and returns a Cellerator-owned native handle. The
Torch adapter accepts that handle with `MechanismModule.from_handle(handle)` or
`SharedSupportRelation.from_handle(spec, handle)` and provides framework
integration around the same owner.

Installed consumers should locate artifacts through the Python distribution
metadata and imported module paths. Do not set development-tree library loader
variables to make an installed package work.
