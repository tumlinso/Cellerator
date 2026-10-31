# Optional C++20 NN module

`cellerator.nn` is an optional host C++20 import facade over the ordinary
`<Cellerator/nn.hh>` API. The module includes the canonical header in its global
module fragment and exports selected `using` declarations for those existing
types and functions. Header, module, and CUDA translation units therefore use
the same declarations, layouts, and ABI. CUDA sources continue to include the
ordinary headers; NVCC does not parse module syntax.

This is a separate add-on package so the main Cellerator build can retain its
GNU/NVCC toolchain. Build it with CMake 3.28 or newer, Ninja, and a Clang
compiler with a matching `clang-scan-deps` executable. The module package
requires an installed Cellerator package exporting `Cellerator::nn`.

The package installs the module source and CMake file-set metadata. CMake
re-scans the installed interface source while building an importing consumer,
so it generates the BMI with that consumer's compiler instead of requiring a
compiler-version-specific BMI from the package.

The bounded external qualification uses CMake 3.28, Clang 18, Ninja, and
`clang-scan-deps` 18. Its NVCC bridge used NVCC 12.9 with GNU 12 as the host
compiler. On CMake 3.28/Ninja, separate executable targets importing the same
installed module target generated a duplicate Ninja rule; place importing
translation units in one target or verify this behavior with another CMake
version. The tested package does not launch CUDA code.
