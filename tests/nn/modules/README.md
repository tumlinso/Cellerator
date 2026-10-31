# Native NN module consumer qualification

`import_consumer.cc` is an installed external consumer, not a constant-only
module check. It maps typed biological feature identities to model indices,
runs a feature encoder and a user-defined two-layer native model, computes an
autograd loss, and steps an ordinary Torch C++ SGD optimizer. A separate
translation unit includes `<Cellerator/nn.hh>`. With
`CELLERATOR_NN_MODULES_TEST_CUDA=ON`, a third translation unit is compiled by
NVCC 12.9 with GNU 12 as its host compiler. Both header TUs accept the same
`Parameters`, `at::Tensor`, and `at::Device` values as the importing TU; they do
not import modules or launch a kernel. In a CUDA-enabled Cellerator package, the
consumer also resolves an existing NN CUDA operation symbol by address without
calling it.

Build and install the main Cellerator package first. Then run:

```sh
tests/nn/modules/run_qualification.sh <cellerator-install-prefix> /tmp/cellerator-nn-module
```

The script builds and installs the optional module add-on, then configures an
external consumer against only the two install prefixes. It requires CMake
3.28+, Ninja, Clang 18, and the matching `clang-scan-deps`; set
`CELLERATOR_MODULE_NINJA` and `CELLERATOR_MODULE_SCANNER` if those tools are not
on `PATH`. Set `CELLERATOR_TORCH_DIR` when Torch is installed outside CMake's
default package search path. To compile the no-launch NVCC bridge, configure a
CUDA-enabled main Cellerator package and set
`CELLERATOR_NN_MODULES_TEST_CUDA=ON`. The default CUDA compiler and host
compiler are the established NVCC 12.9 and GNU 12 pair; override the
`CELLERATOR_CUDA_COMPILER` and `CELLERATOR_CUDA_HOST_COMPILER` variables only
when the matching local installation uses different paths. Set
`CELLERATOR_CUDA_TOOLKIT_ROOT` if Torch's CMake package needs its toolkit
headers/libraries even for the CPU-only module consumer. The script derives
Torch's runtime library directory from `CELLERATOR_TORCH_DIR`; set
`CELLERATOR_TORCH_LIB_DIR` when the library path needs an explicit override.
