# CelleraTorch forward program operation

CE-LIVE-41 provides the thin C++ binding seam that the CE-LIVE-43 fan-in will
register as a Torch custom operation. It consumes the frozen
`cellerator-cellera-torch-entry-v1` executable-program contract; it does not
extend that native ABI.

The caller supplies an already prepared `executable_program`, one native launch
template, and caller-owned CUDA input/output tensors. The launch template keeps
all biological axes, structure identity and epoch, current value generation and
readiness, scalar bindings, workspace, and output-effect contracts in native
Cellerator form. The wrapper validates tensor device, dtype, rank, shape, and
strides, rebinds only the two dense data pointers, obtains the current Torch
CUDA stream for that device, and calls `run_executable_program`.

The wrapper performs no allocation, transfer, format conversion,
canonicalization, planning, preparation, device selection, stream creation, or
synchronization. It returns native execution metadata unchanged, including the
selected candidate/projection, output order, structure epoch, consumed value
generation, and completion stream. Native failures such as stale structure,
unready generation, insufficient workspace, and unsupported width remain
native failures rather than being hidden or reinterpreted by Torch.

Both tensors and the native program remain caller-owned. Tensor pointers are
launch bindings, not biological identity, and may change between calls without
rebuilding prepared state. The wrapper supports exactly one dense input and one
dense output; broader graphs, autograd, parameter ownership, native views,
custom-op registration, and package/build integration belong to their declared
Wave D owners.

The focused test uses a native-run test double to prove current-stream binding,
zero-copy pointer rebinding, repeated prepared execution, output-order and
generation metadata propagation, and rejection of CPU, dtype, rank, shape,
stride, device, lifetime/null, and native-readiness failures. It does not claim
new CUDA kernel or performance evidence.

Until CE-LIVE-43 owns shared CMake registration, reproduce the leaf build with:

```bash
nvcc -std=c++17 -O0 -g -arch=sm_70 \
  -D_GLIBCXX_USE_CXX11_ABI=1 \
  -Icomponents/CelleraTorch/include -Iinclude \
  -Icomponents/CellPack/include -I/usr/local/include \
  -I/usr/local/include/torch/csrc/api/include \
  components/CelleraTorch/src/program_ops.cu \
  components/CelleraTorch/tests/program_ops_test.cc \
  -L/usr/local/lib -Xlinker=-rpath -Xlinker=/usr/local/lib \
  -Xlinker=--no-as-needed \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
  -o /tmp/celleraTorchProgramOpsTest
/tmp/celleraTorchProgramOpsTest
```

The focused V100 correctness controller evidence is
`6c0ff01b-9a97-4781-967d-4f235b064637`; memcheck evidence is
`4aa3f8a1-eb98-47fd-aec0-4cb0a32423cd`. The initial failed controller record
`7f9af755-96c5-4de3-906b-e9358f199ffc` documents the corrected linker
environment (`libtorch_cuda` must not be discarded by `--as-needed`).
