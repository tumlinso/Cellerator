# Python bindings

The Python distribution is named `cellerator`. It provides Torch-free identity
and mechanism declarations plus native numerical entry points. Torch support is
an optional adapter in `cellerator.torch`; it does not create another topology
or parameter owner. No compiler API is exposed through this binding surface.

The resident CUDA bindings are available from `cellerator.cuda` when the
Python/CUDA native capability is built. This module remains Torch-free and owns
the resident streams, buffers, prepared CSR values, and native FP32 kernels.
The optional `cellerator.torch.cuda` adapter adds only tensor validation,
storage borrowing, allocator stream recording, and DLPack conversion; it
delegates every numeric operation to that same native owner. It does not require
the separate libtorch `_torch` extension.

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

The resident CUDA API itself does not require the Torch build option. To use
the Python-facing native CUDA API and its optional thin Torch view, enable the
Python and CUDA capabilities while leaving `CELLERATOR_ENABLE_TORCH` off; then
install/import Torch separately and use `cellerator.torch.cuda`. Borrowed
Torch tensors must be contiguous CUDA FP32 tensors without `requires_grad`.
Adapter calls must use PyTorch's current stream for that device, and borrowed
storage is recorded on that stream before native work is queued. This protects
the allocator lifetime; callers remain responsible for data readiness and
exclusive mutation ordering. `as_tensor(native_buffer)` creates a zero-copy
DLPack alias and transfers the native owner lease into Torch.

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

The resident CUDA surface is separate from the mechanism APIs. A
`cellerator.cuda.Stream(device)` owns a native stream; `Stream.borrow(device,
handle, owner)` wraps an external stream while retaining its owner. `Buffer(shape,
stream)` allocates resident FP32 storage, and `Buffer.borrow(ptr, shape,
capacity_bytes, stream, owner)` creates a caller-owned view. The declared
capacity is a trusted caller precondition: it must describe the accessible
allocation extent. The binding checks that the view fits the declared capacity
and that the pointer is on the selected CUDA device, but cannot prove an
arbitrary caller's extent claim. Keep the owner and borrowed view alive through
completion of queued native work; arithmetic calls enqueue work without
retaining borrowed views after return. A prepared CSR owner retains the buffer
published as its current weights, while its per-call input and output views
remain caller-managed through completion. `upload()` and `download()` are
explicit blocking host-transfer boundaries. `multiply_into` and `axpby_into`
enqueue the native FP32 kernels on the supplied stream.
`PreparedCsr(indptr, indices, weights, source_count, feature_width, stream)`
owns one prepared CSR topology and its current weight generation; `apply_into`
uses caller-provided input/output buffers, and `publish_values` installs a new
weight generation. Its CUDA workspaces and topology stay native-owned.

For Torch programs, import `from cellerator.torch import cuda`. The adapter's
`current_stream(device)` wraps PyTorch's current stream, while `borrow(tensor,
stream)` creates a native view of existing contiguous CUDA FP32 storage.
`multiply_into(a, b, out, stream)` and `axpby_into(alpha, a, beta, b, out,
stream)` dispatch to the same native kernels. `PreparedCsr` accepts CPU NumPy
`uint64` CSR arrays and a Torch FP32 weight tensor, then applies into Torch
input/output tensors or publishes replacement weights through the same native
owner. `as_tensor(native_buffer)` imports a native-owned buffer through DLPack
without copying and uses DLPack's event handoff when the Torch consumer stream
differs from the native producer stream. `record_stream` protects allocator
lifetime; callers remain responsible for producer readiness and exclusive
mutation ordering across streams. The arithmetic remains native FP32
round-to-nearest; the existing prepared-relation candidate permits FMA and
reassociation under its numeric contract. The adapter adds no precision mode.
These operations are forward-only and reject `requires_grad` tensors.

Installed consumers should locate artifacts through the Python distribution
metadata and imported module paths. Do not set development-tree library loader
variables to make an installed package work.
